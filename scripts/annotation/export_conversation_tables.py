"""Export chat messages and annotations into per-participant tables.

This script scans parsed chat transcripts beneath a root directory
(``--transcripts-root``, default ``transcripts_de_ided``) and combines them
with classifier outputs written by ``scripts/annotation/classify_chats.py``
under an annotation outputs root (``--annotation-root``, default
``annotation_outputs``).

For each selected participant, it writes CSV files mirroring the transcript
directory structure beneath ``--output-root``. Each row represents a single
message with:

* Conversation metadata (key, index, creation time).
* Message metadata (index, role, content, timestamp).
* Participant identifier and transcript path.
* A Boolean column per annotation id indicating presence (1), absence (0),
  or no label (empty) using a global score threshold.
* Paths to any JSONL files that supplied annotations for the row.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from analysis_utils.transcripts import normalise_message_fields
from annotation.io import parse_message_indices
from chat import iter_loaded_chats, resolve_bucket_and_rel_path
from utils.cli import add_log_level_argument
from utils.io import get_default_transcripts_root
from utils.utils import pick_latest_per_parent


@dataclass(frozen=True)
class MessageKey:
    """Stable key that identifies a single message in a transcript."""

    participant: str
    source_path: str
    chat_index: int
    message_index: int


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Export chat transcripts and combined annotation outputs into "
            "per-participant CSV tables."
        )
    )
    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=get_default_transcripts_root(),
        help="Root directory containing parsed chat JSON transcripts.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("annotation_outputs"),
        help="Root directory containing JSONL annotation outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("conversation_tables"),
        help=(
            "Directory where per-participant tables will be written. "
            "Each participant gets a subdirectory mirroring transcripts_root."
        ),
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help=(
            "Restrict export to these participant ids (repeatable). "
            "Defaults to all participants discovered under transcripts_root."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=5,
        help=(
            "Minimum annotation score (0-10) required to mark an annotation "
            "as present for a message when annotations are included. Scores "
            "below this threshold are treated as absent."
        ),
    )
    parser.add_argument(
        "--include-annotations",
        action="store_true",
        help=(
            "Include annotation outputs in the tables. When omitted, only "
            "raw transcript metadata is exported."
        ),
    )
    parser.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symbolic links while scanning transcript JSON files.",
    )
    add_log_level_argument(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the export script.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional sequence of command-line arguments. Defaults to ``sys.argv``
        semantics when not provided.

    Returns
    -------
    argparse.Namespace
        Parsed arguments populated with defaults.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.min_score < 0 or args.min_score > 10:
        parser.error("--min-score must be between 0 and 10")
    return args


def _participant_filter(
    participants: Optional[Sequence[str]],
) -> Optional[Set[str]]:
    """Return a normalized participant filter set or None.

    Parameters
    ----------
    participants : Optional[Sequence[str]]
        Optional participant identifiers from the CLI.

    Returns
    -------
    Optional[Set[str]]
        Lowercased participant identifiers when provided, otherwise None.
    """

    if not participants:
        return None
    return {name.strip().lower() for name in participants if name.strip()}


def _collect_transcript_rows(
    transcripts_root: Path,
    participants_filter: Optional[Set[str]],
    *,
    followlinks: bool,
) -> Tuple[
    List[Tuple[MessageKey, Dict[str, object]]],
    Dict[MessageKey, Dict[str, object]],
]:
    """Return ordered transcript rows indexed by message key.

    Parameters
    ----------
    transcripts_root : Path
        Root directory containing parsed chat JSON transcripts.
    participants_filter : Optional[Set[str]]
        Optional set of participant identifiers to include (case-insensitive).
    followlinks : bool
        When True, follow symbolic links while scanning transcripts_root.

    Returns
    -------
    Tuple[List[Tuple[MessageKey, Dict[str, object]]], Dict[MessageKey, Dict[str, object]]]
        A tuple containing:
        - An ordered list of (MessageKey, row_dict) pairs preserving the
          original transcript order across files and messages.
        - A mapping from MessageKey to the shared row_dict for later mutation.
    """

    root = transcripts_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logging.error("Transcripts root not found: %s", root)
        return [], {}

    ordered_rows: List[Tuple[MessageKey, Dict[str, object]]] = []
    index: Dict[MessageKey, Dict[str, object]] = {}

    for key, row in _iter_transcript_message_rows(
        root, participants_filter, followlinks=followlinks
    ):
        if key in index:
            logging.warning(
                "Duplicate message key for participant=%s path=%s "
                "chat_index=%s message_index=%s; keeping first instance",
                key.participant,
                key.source_path,
                key.chat_index,
                key.message_index,
            )
            continue
        ordered_rows.append((key, row))
        index[key] = row

    return ordered_rows, index


def _iter_transcript_message_rows(
    root: Path,
    participants_filter: Optional[Set[str]],
    *,
    followlinks: bool,
) -> Iterator[Tuple[MessageKey, Dict[str, object]]]:
    """Yield (MessageKey, row_dict) pairs for all transcript messages."""

    for file_path, chats in iter_loaded_chats(root, followlinks=followlinks):
        participant, rel_path = _resolve_participant_for_file(
            file_path=file_path,
            root=root,
            participants_filter=participants_filter,
        )
        if participant is None or rel_path is None:
            continue
        for chat_index, chat in enumerate(chats):
            for message_index, message in enumerate(chat.messages):
                message_key = MessageKey(
                    participant=participant,
                    source_path=str(rel_path),
                    chat_index=chat_index,
                    message_index=message_index,
                )
                row_data = _build_row_for_message(
                    key=message_key,
                    chat_key=chat.key,
                    chat_date=chat.date_label,
                    message=message,
                )
                if row_data is None:
                    continue
                yield row_data


def _resolve_participant_for_file(
    *,
    file_path: Path,
    root: Path,
    participants_filter: Optional[Set[str]],
) -> Tuple[Optional[str], Optional[Path]]:
    """Return participant id and relative path for a transcript file."""

    bucket, rel_path = resolve_bucket_and_rel_path(file_path, root)
    if not bucket:
        return None, None
    participant = bucket
    if participants_filter and participant.lower() not in participants_filter:
        return None, None
    return participant, rel_path


def _build_row_for_message(
    *,
    key: MessageKey,
    chat_key: str,
    chat_date: Optional[str],
    message: Mapping[str, object],
) -> Optional[Tuple[MessageKey, Dict[str, object]]]:
    """Return a key and row dict for a single message or None to skip."""

    fields = normalise_message_fields(message)
    if fields is None:
        return None

    row: Dict[str, object] = {
        "participant": key.participant,
        "transcript_rel_path": key.source_path,
        "conversation_key": chat_key,
        "conversation_index": key.chat_index,
        "conversation_date": chat_date,
        "message_index": key.message_index,
        "role": fields["role"],
        "content": fields["content"],
        "message_timestamp": fields["timestamp"],
        "annotation_output_files": "",
    }
    return key, row


def _iter_annotation_files(annotation_root: Path) -> Iterable[Path]:
    """Yield JSONL files beneath the annotation root in sorted order.

    Parameters
    ----------
    annotation_root : Path
        Root directory containing annotation JSONL files.

    Yields
    ------
    Path
        Paths to JSONL files sorted by their string representation.
    """

    root = annotation_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logging.warning("Annotation root not found or not a directory: %s", root)
        return

    candidates: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".jsonl"):
                candidates.append(Path(dirpath) / name)

    if not candidates:
        return

    selected = pick_latest_per_parent(candidates)
    if len(selected) < len(candidates):
        logging.warning(
            "Multiple annotation JSONL files detected under %s; "
            "using only the most recent file per directory and ignoring "
            "earlier runs.",
            root,
        )

    yield from sorted(selected, key=lambda p: str(p).lower())


def _collect_annotations(
    annotation_root: Path,
    *,
    participants_filter: Optional[Set[str]],
    known_keys: Mapping[MessageKey, Mapping[str, object]],
) -> Tuple[
    Dict[MessageKey, Dict[str, int]],
    Dict[MessageKey, Set[str]],
    Set[str],
]:
    """Return per-message annotation scores and contributing files.

    Parameters
    ----------
    annotation_root : Path
        Root directory containing annotation JSONL files.
    participants_filter : Optional[Set[str]]
        Optional set of participant identifiers to include (case-insensitive).
    known_keys : Mapping[MessageKey, Mapping[str, object]]
        Message keys discovered from transcripts; annotations for keys that do
        not exist here are ignored.

    Returns
    -------
    Tuple[
        Dict[MessageKey, Dict[str, int]],
        Dict[MessageKey, Set[str]],
        Set[str],
    ]
        A tuple containing:
        - Mapping from MessageKey to a mapping of annotation_id to max score.
        - Mapping from MessageKey to a set of JSONL paths (as strings) that
          supplied annotation records.
        - Set of all annotation ids observed across processed files.
    """

    score_by_key: Dict[MessageKey, Dict[str, int]] = {}
    files_by_key: Dict[MessageKey, Set[str]] = {}
    all_annotation_ids: Set[str] = set()

    root = annotation_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logging.warning("Annotation root not found: %s", root)
        return score_by_key, files_by_key, all_annotation_ids

    repo_root = Path.cwd()

    for jsonl_path in _iter_annotation_files(root):
        rel_jsonl = str(jsonl_path.relative_to(repo_root))
        try:
            _update_annotation_maps_from_file(
                jsonl_path=jsonl_path,
                rel_jsonl=rel_jsonl,
                participants_filter=participants_filter,
                known_keys=known_keys,
                score_by_key=score_by_key,
                files_by_key=files_by_key,
                all_annotation_ids=all_annotation_ids,
            )
        except OSError as err:
            logging.warning("Failed to read annotation file %s: %s", jsonl_path, err)

    return score_by_key, files_by_key, all_annotation_ids


def _apply_annotations_to_rows(
    ordered_rows: List[Tuple[MessageKey, Dict[str, object]]],
    score_by_key: Mapping[MessageKey, Mapping[str, int]],
    files_by_key: Mapping[MessageKey, Set[str]],
    annotation_ids: Sequence[str],
    *,
    min_score: int,
) -> None:
    """Mutate row dictionaries with annotation presence flags and file links.

    Parameters
    ----------
    ordered_rows : List[Tuple[MessageKey, Dict[str, object]]]
        Ordered (MessageKey, row_dict) pairs for all transcript messages.
    score_by_key : Mapping[MessageKey, Mapping[str, int]]
        Per-message mapping of annotation_id to max observed score.
    files_by_key : Mapping[MessageKey, Set[str]]
        Per-message mapping to sets of JSONL paths that supplied annotations.
    annotation_ids : Sequence[str]
        All annotation identifiers to materialize as columns.
    min_score : int
        Minimum score required to mark an annotation as present.
    """

    sorted_ids = sorted(set(annotation_ids))

    for key, row in ordered_rows:
        key_scores = score_by_key.get(key, {})
        for ann_id in sorted_ids:
            score_value = key_scores.get(ann_id)
            if score_value is None:
                # Annotation never applied to this message.
                row[ann_id] = ""
            else:
                row[ann_id] = 1 if score_value >= min_score else 0

        files = files_by_key.get(key)
        if files:
            row["annotation_output_files"] = ";".join(sorted(files))
        else:
            row["annotation_output_files"] = ""


def _update_annotation_maps_from_file(
    *,
    jsonl_path: Path,
    rel_jsonl: str,
    participants_filter: Optional[Set[str]],
    known_keys: Mapping[MessageKey, Mapping[str, object]],
    score_by_key: Dict[MessageKey, Dict[str, int]],
    files_by_key: Dict[MessageKey, Set[str]],
    all_annotation_ids: Set[str],
) -> None:
    """Update in-memory maps with contents from a single JSONL file."""

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
        if not first_line:
            return
        meta = _parse_meta_line(first_line)
        if meta is not None:
            _collect_annotation_ids_from_meta(meta, all_annotation_ids)

        for line in handle:
            _process_annotation_record_line(
                line=line,
                participants_filter=participants_filter,
                known_keys=known_keys,
                rel_jsonl=rel_jsonl,
                score_by_key=score_by_key,
                files_by_key=files_by_key,
                all_annotation_ids=all_annotation_ids,
            )


def _parse_meta_line(line: str) -> Optional[Mapping[str, object]]:
    """Return meta dictionary from the first JSONL line when available."""

    try:
        meta = json.loads(line)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _collect_annotation_ids_from_meta(
    meta: Mapping[str, object],
    all_annotation_ids: Set[str],
) -> None:
    """Populate annotation ids from a JSONL meta line."""

    snapshots = meta.get("annotation_snapshots")
    if not isinstance(snapshots, dict):
        return
    for ann_id in snapshots.keys():
        if isinstance(ann_id, str) and ann_id:
            all_annotation_ids.add(ann_id)


def _process_annotation_record_line(
    *,
    line: str,
    participants_filter: Optional[Set[str]],
    known_keys: Mapping[MessageKey, Mapping[str, object]],
    rel_jsonl: str,
    score_by_key: Dict[MessageKey, Dict[str, int]],
    files_by_key: Dict[MessageKey, Set[str]],
    all_annotation_ids: Set[str],
) -> None:
    """Update maps based on a single non-meta JSONL data line."""

    stripped = line.strip()
    if not stripped:
        return
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return
    if not isinstance(obj, dict):
        return
    if obj.get("type") == "meta":
        return

    participant_raw = obj.get("participant") or obj.get("ppt_id") or ""
    participant = str(participant_raw).strip()
    if not participant:
        return
    if participants_filter and participant.lower() not in participants_filter:
        return

    source_path_value = obj.get("source_path") or ""
    source_path = str(source_path_value).strip()
    if not source_path:
        return

    index_pair = parse_message_indices(obj)
    if index_pair is None:
        return
    chat_index, message_index = index_pair

    annotation_id_value = obj.get("annotation_id") or ""
    annotation_id = str(annotation_id_value).strip()
    if not annotation_id:
        return
    all_annotation_ids.add(annotation_id)

    score_val = obj.get("score")
    if not isinstance(score_val, (int, float)):
        return
    score_int = int(round(float(score_val)))

    key = MessageKey(
        participant=participant,
        source_path=source_path,
        chat_index=chat_index,
        message_index=message_index,
    )
    if key not in known_keys:
        # Annotation from a different input root or run; skip.
        return

    ann_scores = score_by_key.setdefault(key, {})
    prev_score = ann_scores.get(annotation_id)
    if prev_score is not None:
        logging.warning(
            "Duplicate annotation record for participant=%s path=%s "
            "chat_index=%s message_index=%s annotation_id=%s; "
            "keeping the first score (%s) and ignoring subsequent value (%s) "
            "from %s",
            participant,
            source_path,
            chat_index,
            message_index,
            annotation_id,
            prev_score,
            score_int,
            rel_jsonl,
        )
        return
    ann_scores[annotation_id] = score_int

    files = files_by_key.setdefault(key, set())
    files.add(rel_jsonl)


def _group_rows_by_participant_and_path(
    ordered_rows: List[Tuple[MessageKey, Dict[str, object]]],
) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    """Return rows grouped by (participant, transcript_rel_path).

    Parameters
    ----------
    ordered_rows : List[Tuple[MessageKey, Dict[str, object]]]
        Ordered (MessageKey, row_dict) pairs for all transcript messages.

    Returns
    -------
    Dict[Tuple[str, str], List[Dict[str, object]]]
        Mapping from (participant, transcript_rel_path) to an ordered list of
        row dictionaries for that file.
    """

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for key, row in ordered_rows:
        transcript_rel_path = str(row.get("transcript_rel_path"))
        grouped_key = (key.participant, transcript_rel_path)
        grouped.setdefault(grouped_key, []).append(row)
    return grouped


def _derive_csv_headers(
    annotation_ids: Sequence[str],
) -> List[str]:
    """Return ordered CSV headers with annotation columns appended.

    Parameters
    ----------
    annotation_ids : Sequence[str]
        Annotation identifiers to use as column names.

    Returns
    -------
    List[str]
        Ordered header names for CSV files.
    """

    base_headers = [
        "participant",
        "transcript_rel_path",
        "conversation_key",
        "conversation_index",
        "conversation_date",
        "message_index",
        "role",
        "content",
        "message_timestamp",
        "annotation_output_files",
    ]
    annotation_headers = sorted(set(annotation_ids))
    return base_headers + annotation_headers


def _write_csv_tables(
    output_root: Path,
    grouped_rows: Mapping[Tuple[str, str], List[Mapping[str, object]]],
    headers: Sequence[str],
) -> None:
    """Write grouped rows to per-participant CSV files.

    Parameters
    ----------
    output_root : Path
        Root directory for generated tables.
    grouped_rows : Mapping[Tuple[str, str], List[Mapping[str, object]]]
        Mapping from (participant, transcript_rel_path) to ordered row dicts.
    headers : Sequence[str]
        CSV header names used for all tables.
    """

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for (_participant, rel_path_str), rows in grouped_rows.items():
        rel_path = Path(rel_path_str)

        # Mirror the transcript subdirectory layout beneath the output root,
        # preserving buckets such as "under_irb/irb_05".
        parent_dir = root / rel_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Derive an output filename that is clearly tabular but preserves the
        # original JSON source basename.
        base_name = rel_path.name
        output_name = f"{base_name}.csv"
        output_path = parent_dir / output_name

        try:
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(headers),
                    quoting=csv.QUOTE_ALL,
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except OSError as err:
            logging.error("Failed to write CSV %s: %s", output_path, err)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for exporting per-participant chat tables.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional sequence of command-line arguments.

    Returns
    -------
    int
        Process exit status code (0 on success, non-zero on error).
    """

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    participants_filter = _participant_filter(args.participants)

    logging.info("Collecting transcript rows from %s", args.transcripts_root)
    ordered_rows, index = _collect_transcript_rows(
        args.transcripts_root,
        participants_filter,
        followlinks=bool(args.follow_links),
    )
    if not ordered_rows:
        logging.warning("No transcript messages found to export.")
        return 0

    all_annotation_ids: Set[str] = set()
    score_by_key: Dict[MessageKey, Dict[str, int]] = {}
    files_by_key: Dict[MessageKey, Set[str]] = {}

    if args.include_annotations:
        logging.info("Collecting annotations from %s", args.annotation_root)
        score_by_key, files_by_key, all_annotation_ids = _collect_annotations(
            args.annotation_root,
            participants_filter=participants_filter,
            known_keys=index,
        )

        if not all_annotation_ids:
            logging.warning(
                "No annotation ids discovered under %s; tables will contain only "
                "transcript metadata.",
                args.annotation_root,
            )
        else:
            logging.info(
                "Applying %d annotation ids with min_score=%d",
                len(all_annotation_ids),
                args.min_score,
            )
            _apply_annotations_to_rows(
                ordered_rows,
                score_by_key=score_by_key,
                files_by_key=files_by_key,
                annotation_ids=sorted(all_annotation_ids),
                min_score=args.min_score,
            )
    else:
        logging.info(
            "Skipping annotation outputs. Re-run with --include-annotations "
            "to add annotation presence columns."
        )

    grouped_rows = _group_rows_by_participant_and_path(ordered_rows)
    headers = _derive_csv_headers(sorted(all_annotation_ids))

    logging.info(
        "Writing tables for %d (participant, transcript) groups into %s",
        len(grouped_rows),
        args.output_root,
    )
    _write_csv_tables(args.output_root, grouped_rows, headers)

    logging.info("Export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
