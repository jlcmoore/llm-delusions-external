"""
Prepare manual-annotation datasets aligned with existing classification outputs.

This script derives manual datasets from already-computed LLM classification
JSONL files under ``annotation_outputs/``. Each dataset item corresponds to a
specific classification record, so downstream agreement analyses have perfect
coverage (no "Missing" entries) for the chosen model.

For each selected classification item, the script:

* Reuses the target message content and metadata from the JSONL record.
* Applies optional sampling, participant, and annotation filters.
* Writes a JSON Lines manual dataset that ``analysis/viewer/manual_annotator.html``
  can consume. Preceding context messages are not reloaded from
  ``transcripts_de_ided``; the ``preceding`` field in each item is left empty.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

from annotation.configs import parse_annotation_scope, resolve_annotation
from annotation.cutoffs import load_llm_cutoffs_from_json
from utils.cli import add_participants_argument, add_randomize_per_ppt_argument
from utils.io import (
    get_default_transcripts_root,
    iter_jsonl_dicts,
    iter_objects_with_location,
)
from utils.utils import allocate_per_participant

_TRANSCRIPTS_TABLE_CACHE: Dict[Path, pd.DataFrame] = {}


def _load_transcripts_table(transcripts_path: Path) -> Optional[pd.DataFrame]:
    """Return a cached transcripts table for the given path.

    Parameters
    ----------
    transcripts_path:
        Path to the transcripts Parquet file containing one row per
        message, including content.

    Returns
    -------
    Optional[pandas.DataFrame]
        Cached transcripts table when loading succeeds; otherwise ``None``
        after logging a diagnostic.
    """

    resolved = transcripts_path.expanduser().resolve()
    cached = _TRANSCRIPTS_TABLE_CACHE.get(resolved)
    if cached is not None:
        return cached

    try:
        frame = pd.read_parquet(resolved)
    except (OSError, ValueError) as err:
        logging.error("Failed to read transcripts table %s: %s", resolved, err)
        return None

    if frame.empty:
        logging.warning("Transcripts table %s is empty.", resolved)
        _TRANSCRIPTS_TABLE_CACHE[resolved] = frame
        return frame

    _TRANSCRIPTS_TABLE_CACHE[resolved] = frame
    return frame


def _read_classification_items(path: Path) -> Iterable[dict]:
    """Yield non-meta JSON objects from a classification JSONL file."""

    try:
        for obj in iter_jsonl_dicts(path):
            if obj.get("type") == "meta":
                continue
            yield obj
    except OSError as err:
        logging.error("Failed to read classification file %s: %s", path, err)


def _build_output_path(
    *, model: str, annotation_id: Optional[str], args: argparse.Namespace
) -> Path:
    """Return an output path under manual_annotation_inputs/."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("manual_annotation_inputs").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frag_parts: List[str] = [f"model={model}"]
    if annotation_id:
        frag_parts.append(f"annotation={annotation_id}")
    if getattr(args, "max_items", None):
        frag_parts.append(f"max-items={args.max_items}")
    if getattr(args, "preceding_context", None) is not None:
        frag_parts.append(f"preceding-context={args.preceding_context}")
    cutoffs_json = getattr(args, "llm_cutoffs_json", None)
    use_positive_only = bool(getattr(args, "llm_positive_only", False))
    if cutoffs_json and use_positive_only:
        cutoffs_name = Path(str(cutoffs_json)).name
        frag_parts.append(f"llm-cutoffs-json={cutoffs_name}")
    elif use_positive_only and getattr(args, "llm_score_cutoff", None) is not None:
        frag_parts.append(f"llm-score-cutoff={args.llm_score_cutoff}")
    fragment = "&".join(frag_parts)
    filename = f"{timestamp}__{fragment}.jsonl"
    return root / filename


def _cap_items_per_annotation(
    items: Sequence[dict],
    *,
    max_items: int,
    randomize: bool,
    random_seed: Optional[int],
) -> List[dict]:
    """Return items with at most ``max_items`` per annotation id.

    Parameters
    ----------
    items:
        Sequence of manual-annotation item dictionaries.
    max_items:
        Maximum number of items to retain for any single annotation id.
        When zero or negative, no additional per-annotation caps are
        applied and the original sequence is returned.
    randomize:
        When True, select retained items uniformly at random for
        annotation ids whose counts exceed ``max_items``. When False,
        keep the first ``max_items`` items in sequence order.
    random_seed:
        Optional random seed used when ``randomize`` is True.

    Returns
    -------
    List[dict]
        New list of items with per-annotation limits enforced.
    """

    if max_items <= 0:
        return list(items)

    index_by_annotation: Dict[str, List[int]] = {}
    for index, item in enumerate(items):
        annotation_text = str(item.get("annotation_id") or "").strip()
        if not annotation_text:
            continue
        bucket = index_by_annotation.setdefault(annotation_text, [])
        bucket.append(index)

    if not index_by_annotation:
        return list(items)

    rng: Optional[random.Random] = None
    if randomize:
        rng = random.Random(random_seed)

    keep_indices: Set[int] = set()
    for annotation_text, index_list in index_by_annotation.items():
        if len(index_list) <= max_items:
            keep_indices.update(index_list)
            continue
        if randomize and rng is not None:
            selected = set(rng.sample(index_list, max_items))
        else:
            selected = set(index_list[:max_items])
        keep_indices.update(selected)

    capped_items: List[dict] = []
    for index, item in enumerate(items):
        annotation_text = str(item.get("annotation_id") or "").strip()
        if not annotation_text:
            capped_items.append(item)
            continue
        if index in keep_indices:
            capped_items.append(item)

    return capped_items


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for classification-aligned datasets."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare manual-annotation datasets aligned with existing "
            "classification outputs under annotation_outputs/."
        )
    )
    parser.add_argument(
        "--outputs-root",
        default="annotation_outputs",
        help="Root directory containing classification JSONL outputs.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to align with (e.g., gpt-5.1).",
    )
    parser.add_argument(
        "--annotation-id",
        help=(
            "Optional annotation id to restrict items. When omitted, all "
            "annotations present in the selected runs are included."
        ),
    )
    add_participants_argument(
        parser,
        help_text=(
            "Restrict to these participant ids (repeatable). Defaults to all "
            "participants present in the selected runs."
        ),
    )
    parser.add_argument(
        "--preceding-context",
        type=int,
        default=3,
        help=(
            "Number of preceding messages to include as context (default: 3). "
            "Capped at 10 for safety."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help=(
            "Optional maximum number of items to include per annotation id "
            "across all runs. Defaults to 0 (no additional per-code limit). "
            "When combined with --randomize, items are sampled rather than "
            "taken in file order."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional explicit output path. When omitted, a timestamped file "
            "is created under manual_annotation_inputs/."
        ),
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help=(
            "When used with --max-items, randomly sample items for each "
            "annotation id instead of taking the first N in file order."
        ),
    )
    add_randomize_per_ppt_argument(parser)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for --randomize sampling. Defaults to a "
            "non-deterministic seed when omitted."
        ),
    )
    parser.add_argument(
        "--llm-score-cutoff",
        type=int,
        default=None,
        help=(
            "Minimum LLM score (0-10) required to treat a classification as "
            "positive when filtering by --llm-positive-only. When omitted, "
            "only per-annotation cutoffs from --llm-cutoffs-json are used. "
            "When --llm-cutoffs-json is "
            "provided, per-annotation cutoffs from that file take precedence "
            "over this global value."
        ),
    )
    parser.add_argument(
        "--llm-cutoffs-json",
        type=str,
        help=(
            "Optional JSON file containing per-annotation LLM score cutoffs. "
            "This should be the canonical metrics or cutoffs JSON used in the "
            "paper; its per-annotation thresholds are applied when "
            "--llm-positive-only is set."
        ),
    )
    parser.add_argument(
        "--llm-positive-only",
        action="store_true",
        help=(
            "Restrict manual-annotation items to messages that the LLM "
            "classified as positive at or above --llm-score-cutoff. This is "
            "useful for focused inspection of potential false positives."
        ),
    )
    parser.add_argument(
        "--defaults-only",
        action="store_true",
        help=(
            "Restrict classification runs to outputs whose filename ends with "
            "'__defaults.jsonl'."
        ),
    )
    parser.add_argument(
        "--match-arguments-from",
        metavar="JSONL_PATH",
        help=(
            "Restrict classification runs to the job family containing the "
            "given JSONL file, matching the basename-based family discovery "
            "used by analysis scripts (for example, all siblings of "
            "all_annotations.jsonl under the outputs root)."
        ),
    )
    parser.add_argument(
        "--preprocessed-table",
        type=Path,
        help=(
            "Preprocessed per-message annotation table (Parquet) produced by "
            "preprocess_annotation_family.py. When LLM-positive-only filters "
            "are not active, messages are sampled from this table using "
            "random index selection instead of from raw JSONL classification "
            "outputs."
        ),
        required=True,
    )
    parser.add_argument(
        "--transcripts-table",
        type=Path,
        default=Path("transcripts_data/transcripts.parquet"),
        help=(
            "Parquet file containing transcript messages with content. This "
            "is used together with --preprocessed-table to recover message "
            "text and metadata for manual-annotation items."
        ),
    )

    args = parser.parse_args(argv)
    if args.llm_score_cutoff is not None:
        if args.llm_score_cutoff < 0 or args.llm_score_cutoff > 10:
            parser.error("--llm-score-cutoff must be between 0 and 10.")
    return args


def _select_runs(
    *,
    runs: Sequence[object],
    model: str,
    annotation_id: Optional[str],
    participants: Optional[Sequence[str]],
) -> List[object]:
    """Return runs matching the requested model/annotation/participants."""

    selected: List[object] = []
    model_norm = str(model or "").strip()
    participant_set = {p.strip() for p in (participants or []) if p and p.strip()}

    for run in runs:
        if model_norm and run.model != model_norm:
            continue
        if annotation_id and annotation_id not in run.annotation_ids:
            continue
        if participant_set:
            # Require at least one overlapping participant when both sides
            # report participants.
            if run.participants and not participant_set.intersection(
                set(run.participants)
            ):
                continue
        selected.append(run)

    # When no specific annotation id is requested, prefer "base" runs that
    # contain the full annotation set (for example, classification passes that
    # did not filter to a single annotation via an --annotation flag). These
    # are typically encoded without an ``__annotation=...`` fragment in the
    # filename. If no such runs are available, fall back to the full set.
    if not annotation_id and selected:
        base_runs: List[object] = []
        filtered_runs: List[object] = []
        for run in selected:
            name = run.path.name
            if "__annotation=" in name:
                filtered_runs.append(run)
            else:
                base_runs.append(run)
        if base_runs:
            # Include any filtered runs that introduce annotation ids not
            # present in the base runs. This allows per-annotation runs (for
            # example, user-endorses-delusion-only passes) to contribute
            # additional codes that are not covered by the base passes,
            # without duplicating coverage for annotations already present.
            base_annotation_ids: set[str] = set()
            for run in base_runs:
                for ann in run.annotation_ids:
                    if ann:
                        base_annotation_ids.add(ann)
            extra_filtered: List[object] = []
            for run in filtered_runs:
                if any(
                    ann and ann not in base_annotation_ids for ann in run.annotation_ids
                ):
                    extra_filtered.append(run)
            return base_runs + extra_filtered

    return selected


def _lookup_transcript_row(
    transcripts_path: Path,
    *,
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
) -> Optional[Dict[str, object]]:
    """Return a transcript row for the given message location.

    Parameters
    ----------
    transcripts_path:
        Path to the transcripts Parquet file produced by the parsing
        pipeline (for example, transcripts_data/transcripts.parquet).
    participant:
        Participant identifier string.
    source_path:
        Relative transcript path within transcripts_de_ided.
    chat_index:
        Zero-based conversation index.
    message_index:
        Zero-based message index within the conversation.

    Returns
    -------
    Optional[Dict[str, object]]
        Dictionary containing role, timestamp, chat_key, chat_date, and
        content fields when a matching row is found; otherwise ``None``
        after logging a diagnostic.
    """

    frame = _load_transcripts_table(transcripts_path)
    if frame is None or frame.empty:
        logging.error(
            "Transcripts table %s is unavailable when looking up %s/%s (%d, %d).",
            transcripts_path,
            participant,
            source_path,
            chat_index,
            message_index,
        )
        return None

    chat_index_int = int(chat_index)
    message_index_int = int(message_index)
    mask = (
        (frame["participant"] == participant)
        & (frame["source_path"] == source_path)
        & (frame["chat_index"] == chat_index_int)
        & (frame["message_index"] == message_index_int)
    )
    filtered = frame[mask]
    if filtered.empty:
        logging.warning(
            "No transcript row found in %s for %s/%s (%d, %d).",
            transcripts_path,
            participant,
            source_path,
            chat_index_int,
            message_index_int,
        )
        return None

    row = filtered.iloc[0]
    return {
        "role": str(row.get("role") or "").strip(),
        "chat_key": row.get("chat_key"),
        "content": row.get("content") or "",
    }


def _build_preceding_from_transcripts(
    transcripts_path: Path,
    *,
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
    preceding_context: int,
) -> List[Dict[str, object]]:
    """Return up to ``preceding_context`` prior messages from transcripts.

    Parameters
    ----------
    transcripts_path:
        Path to the transcripts Parquet file.
    participant:
        Participant identifier string.
    source_path:
        Relative transcript path within transcripts_de_ided.
    chat_index:
        Zero-based conversation index.
    message_index:
        Zero-based message index within the conversation.
    preceding_context:
        Maximum number of preceding messages to include.

    Returns
    -------
    List[Dict[str, object]]
        List of preceding message dictionaries in ascending index order.
    """

    if preceding_context <= 0:
        return []

    frame = _load_transcripts_table(transcripts_path)
    if frame is None or frame.empty:
        logging.error(
            "Transcripts table %s is unavailable when building preceding "
            "context for %s/%s (%d, %d).",
            transcripts_path,
            participant,
            source_path,
            chat_index,
            message_index,
        )
        return []

    chat_index_int = int(chat_index)
    message_index_int = int(message_index)
    lower_bound = max(message_index_int - int(preceding_context), 0)

    mask = (
        (frame["participant"] == participant)
        & (frame["source_path"] == source_path)
        & (frame["chat_index"] == chat_index_int)
        & (frame["message_index"] < message_index_int)
        & (frame["message_index"] >= lower_bound)
    )
    subset = frame[mask]
    if subset.empty:
        return []

    subset = subset.sort_values("message_index")
    preceding: List[Dict[str, object]] = []
    for _, row in subset.iterrows():
        preceding.append(
            {
                "role": str(row.get("role") or "").strip(),
                "chat_key": row.get("chat_key"),
                "content": row.get("content") or "",
            }
        )

    return preceding


def _build_preprocessed_item(
    transcripts_path: Path,
    *,
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
    annotation_id: str,
    allowed_roles: Optional[Set[str]],
    preceding_context: int,
    llm_score: Optional[int] = None,
    llm_positive: Optional[bool] = None,
) -> Optional[Dict[str, object]]:
    """Return a manual-annotation item built from preprocessed metadata.

    This helper loads the focal message and optional preceding context from
    the transcripts table and enforces role-based scope for the annotation.

    Parameters
    ----------
    transcripts_path:
        Path to the transcripts Parquet file.
    participant:
        Participant identifier string.
    source_path:
        Relative transcript path within transcripts_de_ided.
    chat_index:
        Zero-based conversation index.
    message_index:
        Zero-based message index within the conversation.
    annotation_id:
        Identifier of the annotation to attach.
    allowed_roles:
        Optional set of roles for which the annotation is defined.
    preceding_context:
        Maximum number of preceding messages to include.
    llm_score:
        Optional integer LLM score associated with the classification.
    llm_positive:
        Optional positivity flag derived from the score and cutoff.

    Returns
    -------
    Optional[Dict[str, object]]
        Complete manual-annotation record when the message is compatible
        with the annotation scope and transcript metadata can be loaded;
        otherwise ``None``.
    """

    transcript_row = _lookup_transcript_row(
        transcripts_path,
        participant=participant,
        source_path=source_path,
        chat_index=chat_index,
        message_index=message_index,
    )
    if not transcript_row:
        return None

    role_text = str(transcript_row.get("role") or "").strip().lower()
    if allowed_roles:
        if not role_text or role_text not in allowed_roles:
            return None

    preceding_messages = _build_preceding_from_transcripts(
        transcripts_path,
        participant=participant,
        source_path=source_path,
        chat_index=chat_index,
        message_index=message_index,
        preceding_context=preceding_context,
    )

    record: Dict[str, object] = {
        "type": "item",
        "participant": participant,
        "ppt_id": participant,
        "source_path": source_path,
        "chat_index": chat_index,
        "chat_key": transcript_row.get("chat_key"),
        "message_index": message_index,
        "role": transcript_row.get("role"),
        "annotation_id": annotation_id,
        "annotation": annotation_id,
        "content": transcript_row.get("content") or "",
        "preceding": preceding_messages,
    }
    if llm_score is not None:
        record["llm_score"] = llm_score
    if llm_positive is not None:
        record["llm_positive"] = llm_positive

    return record


def _build_item_from_chat_frame(
    chat_frame: pd.DataFrame,
    *,
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
    annotation_id: str,
    allowed_roles: Optional[Set[str]],
    preceding_context: int,
    llm_score: Optional[int] = None,
    llm_positive: Optional[bool] = None,
    transcripts_path: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """Return a manual-annotation item from a per-chat transcripts subset.

    This helper assumes ``chat_frame`` contains all transcript rows for a
    single chat (participant, source path, and chat index) and uses it to
    locate the focal message and preceding context without scanning the full
    transcripts table. It mirrors the structure of ``_build_preprocessed_item``
    so that vectorised joining logic can reuse a single implementation.

    Parameters
    ----------
    chat_frame:
        DataFrame containing all transcript rows for one chat.
    participant:
        Participant identifier string for the chat.
    source_path:
        Relative transcript path within transcripts_de_ided.
    chat_index:
        Zero-based conversation index for the chat.
    message_index:
        Zero-based message index within the conversation.
    annotation_id:
        Identifier of the annotation to attach.
    allowed_roles:
        Optional set of roles for which the annotation is defined.
    preceding_context:
        Maximum number of preceding messages to include.
    llm_score:
        Optional integer LLM score associated with the classification.
    llm_positive:
        Optional positivity flag derived from the score and cutoff.
    transcripts_path:
        Optional path to the transcripts Parquet file, used only for
        diagnostic logging.

    Returns
    -------
    Optional[Dict[str, object]]
        Complete manual-annotation record when the message is compatible
        with the annotation scope and transcript metadata can be loaded;
        otherwise ``None``.
    """

    if chat_frame.empty:
        return None

    focal_rows = chat_frame[chat_frame["message_index"] == int(message_index)]
    if focal_rows.empty:
        if transcripts_path is not None:
            logging.warning(
                "No transcript row found in %s for %s/%s (%d, %d).",
                transcripts_path,
                participant,
                source_path,
                int(chat_index),
                int(message_index),
            )
        return None

    focal = focal_rows.iloc[0]

    role_text = str(focal.get("role") or "").strip().lower()
    if allowed_roles:
        if not role_text or role_text not in allowed_roles:
            return None

    preceding_messages: List[Dict[str, object]] = []
    if preceding_context > 0:
        message_index_int = int(message_index)
        lower_bound = max(message_index_int - int(preceding_context), 0)
        prev_subset = chat_frame[
            (chat_frame["message_index"] < message_index_int)
            & (chat_frame["message_index"] >= lower_bound)
        ]
        if not prev_subset.empty:
            for _, prev in prev_subset.sort_values("message_index").iterrows():
                preceding_messages.append(
                    {
                        "role": str(prev.get("role") or "").strip(),
                        "chat_key": prev.get("chat_key"),
                        "content": prev.get("content") or "",
                    }
                )

    record: Dict[str, object] = {
        "type": "item",
        "participant": participant,
        "ppt_id": participant,
        "source_path": source_path,
        "chat_index": int(chat_index),
        "chat_key": focal.get("chat_key"),
        "message_index": int(message_index),
        "role": focal.get("role"),
        "annotation_id": annotation_id,
        "annotation": annotation_id,
        "content": focal.get("content") or "",
        "preceding": preceding_messages,
    }
    if llm_score is not None:
        record["llm_score"] = llm_score
    if llm_positive is not None:
        record["llm_positive"] = llm_positive

    return record


def _gather_items(
    *,
    runs: Sequence[object],
    annotation_id: Optional[str],
    participants: Optional[Sequence[str]],
    max_items: int,
    randomize: bool,
    randomize_per_ppt: str,
    random_seed: Optional[int],
    llm_score_cutoff: Optional[int],
    llm_positive_only: bool,
    cutoffs_by_id: Optional[Dict[str, int]],
) -> Tuple[List[dict], List[str]]:
    """Collect classification-aligned manual items with context."""

    participant_set = {p.strip() for p in (participants or []) if p and p.strip()}
    seen_keys: Dict[Tuple[str, str, int, int, str], dict] = {}
    items: List[dict] = []

    # Track all annotation ids present in the selected runs so that, when
    # requested, we can synthesize shared-message items for every code even
    # when some annotations are missing at particular message locations.
    all_annotation_ids: Set[str] = set()
    for run in runs:
        for ann in run.annotation_ids:
            ann_clean = str(ann or "").strip()
            if ann_clean:
                all_annotation_ids.add(ann_clean)

    # When shared-message sampling is active, we rely on annotation scope
    # metadata so that synthesized items respect the intended role (for
    # example, user-only versus assistant-only codes). Annotations whose
    # specifications are missing or do not restrict scope are treated as
    # applying to all roles.
    scope_by_annotation_id: Dict[str, Optional[Set[str]]] = {}
    for ann_id in all_annotation_ids:
        try:
            spec = resolve_annotation(ann_id)
        except ValueError:
            scope_by_annotation_id[ann_id] = None
            continue
        allowed_roles = parse_annotation_scope(spec)
        scope_by_annotation_id[ann_id] = allowed_roles

    # When max_items is provided together with randomize and without
    # LLM-positive-only filtering, maintain a streaming reservoir of message
    # locations so that all annotations share the same underlying set of
    # message keys while keeping memory bounded. We sample at the message-key
    # level rather than per-annotation-item so that downstream analyses see a
    # consistent set of chats across codes. Reservoirs are kept separately for
    # each primary role (user versus assistant) so annotations with scoped
    # roles share messages that are in-scope.
    use_key_reservoir = bool(
        max_items and max_items > 0 and randomize and not llm_positive_only
    )
    # For each selected message location we keep a single template record per
    # role; per-annotation items are synthesized later so that every
    # annotation id sees the same underlying set of messages compatible with
    # its scope.
    key_reservoir_by_role: Dict[str, Dict[Tuple[str, str, int, int], dict]] = {}
    key_order_by_role: Dict[str, List[Tuple[str, str, int, int]]] = {}
    total_keys_seen_by_role: Dict[str, int] = {}
    key_rng: Optional[random.Random] = None

    annotation_ids: set[str] = set()
    participants_seen: set[str] = set()
    per_annotation_counts: Dict[str, int] = {}
    raw_annotation_counts: Dict[str, int] = {}
    positive_annotation_counts: Dict[str, int] = {}
    per_annotation_cutoffs: Dict[str, int] = dict(cutoffs_by_id or {})

    with tqdm(
        total=len(runs), desc="Scanning classification runs", unit="run"
    ) as run_bar:
        for run in runs:
            run_bar.update(1)
            with tqdm(
                desc=f"Collecting items from {run.path.name}",
                unit="item",
                leave=False,
            ) as item_bar:
                for (
                    obj,
                    source_path,
                    chat_index,
                    message_index,
                ) in iter_objects_with_location(_read_classification_items(run.path)):
                    item_bar.update(1)
                    ann_id = str(obj.get("annotation_id") or "").strip()
                    if not ann_id:
                        continue
                    if annotation_id and ann_id != annotation_id:
                        continue
                    raw_annotation_counts[ann_id] = (
                        raw_annotation_counts.get(ann_id, 0) + 1
                    )

                    participant = str(obj.get("participant") or "").strip()
                    if not participant:
                        continue
                    if participant_set and participant not in participant_set:
                        continue

                    # Derive an integer LLM score and positivity flag when
                    # available. Missing or malformed scores are treated as
                    # non-positive for filtering purposes. When per-annotation
                    # cutoffs are available, they take precedence over the
                    # global llm_score_cutoff so that manual datasets respect
                    # the canonical operating points from metrics JSON files.
                    raw_score = obj.get("score")
                    llm_score: Optional[int] = None
                    if isinstance(raw_score, (int, float)):
                        try:
                            llm_score = int(round(float(raw_score)))
                        except (TypeError, ValueError, OverflowError):
                            llm_score = None
                    llm_positive: Optional[bool] = None
                    cutoff_for_annotation: Optional[int] = None
                    if ann_id in per_annotation_cutoffs:
                        cutoff_for_annotation = per_annotation_cutoffs[ann_id]
                    else:
                        cutoff_for_annotation = llm_score_cutoff
                    if llm_score is not None and cutoff_for_annotation is not None:
                        llm_positive = llm_score >= cutoff_for_annotation
                    if llm_positive_only and not llm_positive:
                        continue

                    key = (participant, source_path, chat_index, message_index, ann_id)
                    if key in seen_keys:
                        continue

                    # When max_items is provided together with
                    # --llm-positive-only, enforce a per-annotation-id cap in
                    # the streaming loop so that the dataset remains
                    # balanced across codes for the positive-only regime.
                    if (
                        llm_positive_only
                        and max_items
                        and max_items > 0
                        and not randomize
                    ):
                        if per_annotation_counts.get(ann_id, 0) >= max_items:
                            continue

                    # For manual annotation, we now rely directly on the
                    # classification outputs and do not attempt to re-load
                    # transcripts from transcripts_de_ided. The target message
                    # content is taken as-is from the JSONL record and any
                    # stored preceding context is reused when available.
                    previous: List[dict] = []
                    raw_preceding = obj.get("preceding")
                    if isinstance(raw_preceding, list):
                        for prev in raw_preceding:
                            if not isinstance(prev, dict):
                                continue
                            index_raw = prev.get("index")
                            try:
                                index_val = int(index_raw)
                            except (TypeError, ValueError):
                                index_val = index_raw
                            role_text = str(prev.get("role") or "").strip()
                            content_text = str(prev.get("content") or "")
                            timestamp_raw = prev.get("timestamp")
                            timestamp_text = (
                                str(timestamp_raw).strip()
                                if isinstance(timestamp_raw, str)
                                and timestamp_raw.strip()
                                else ""
                            )
                            previous.append(
                                {
                                    "index": index_val,
                                    "role": role_text,
                                    "content": content_text,
                                    "timestamp": timestamp_text,
                                }
                            )

                    record = {
                        "type": "item",
                        "participant": participant,
                        "ppt_id": obj.get("ppt_id") or participant,
                        "source_path": source_path,
                        "chat_index": chat_index,
                        "chat_key": obj.get("chat_key"),
                        "chat_date": obj.get("chat_date"),
                        "message_index": message_index,
                        "role": obj.get("role"),
                        "timestamp": obj.get("timestamp"),
                        "annotation_id": ann_id,
                        "annotation": obj.get("annotation") or ann_id,
                        "content": obj.get("content") or "",
                        "preceding": previous,
                        "llm_score": llm_score,
                        "llm_positive": llm_positive,
                    }

                    seen_keys[key] = record
                    annotation_ids.add(ann_id)
                    participants_seen.add(participant)

                    if use_key_reservoir:
                        # Streaming reservoir sampling at the message-key level
                        # so that all annotations share the same underlying
                        # message locations, tracked separately for user and
                        # assistant roles.
                        if key_rng is None:
                            key_rng = random.Random(random_seed)

                        role_text = str(record.get("role") or "").strip().lower()
                        # Restrict shared-message sampling to core conversational
                        # roles; other roles are excluded from the reservoir.
                        if role_text not in {"user", "assistant"}:
                            continue

                        base_key = (participant, source_path, chat_index, message_index)
                        role_reservoir = key_reservoir_by_role.setdefault(role_text, {})
                        role_order = key_order_by_role.setdefault(role_text, [])
                        if base_key in role_reservoir:
                            # We already have a template for this message and
                            # role; per-annotation variants will be synthesized
                            # later.
                            continue

                        total_seen = total_keys_seen_by_role.get(role_text, 0) + 1
                        total_keys_seen_by_role[role_text] = total_seen
                        if len(role_order) < max_items:
                            role_order.append(base_key)
                            role_reservoir[base_key] = record
                        else:
                            assert key_rng is not None
                            replacement_index = key_rng.randint(0, total_seen - 1)
                            if replacement_index < max_items:
                                evicted_key = role_order[replacement_index]
                                role_reservoir.pop(evicted_key, None)
                                role_order[replacement_index] = base_key
                                role_reservoir[base_key] = record
                        # per_annotation_counts is not used for the key-based
                        # reservoir but remains available for positive-only
                        # diagnostics in other modes.
                    else:
                        items.append(record)
                        if llm_positive_only:
                            per_annotation_counts[ann_id] = (
                                per_annotation_counts.get(ann_id, 0) + 1
                            )
                            positive_annotation_counts[ann_id] = (
                                positive_annotation_counts.get(ann_id, 0) + 1
                            )

    # When key-level streaming is active, synthesize one item per annotation
    # id for each selected message location so that every annotation sees the
    # same underlying set of ``max_items`` messages per role (or fewer when
    # the source data does not provide enough unique locations).
    if use_key_reservoir:
        target_annotation_ids: List[str]
        if annotation_id:
            target_annotation_ids = [annotation_id]
        else:
            # Fall back to the full set of codes advertised by the runs so
            # that even rarely-seen annotations receive items.
            target_annotation_ids = sorted(all_annotation_ids)

        synthesized: List[dict] = []
        for role_text, key_order in key_order_by_role.items():
            role_reservoir = key_reservoir_by_role.get(role_text, {})
            normalized_role = str(role_text or "").strip().lower()
            for base_key in key_order:
                template = role_reservoir.get(base_key)
                if not template:
                    continue
                for ann in target_annotation_ids:
                    allowed_roles = scope_by_annotation_id.get(ann)
                    if allowed_roles:
                        # Restrict synthesized items to messages whose role is
                        # compatible with the annotation's scope.
                        if not normalized_role or normalized_role not in allowed_roles:
                            continue
                    new_record = dict(template)
                    new_record["annotation_id"] = ann
                    new_record["annotation"] = ann
                    # Drop LLM classification-specific fields so that manual
                    # annotators are not influenced by model scores.
                    new_record.pop("llm_score", None)
                    new_record.pop("llm_positive", None)
                    synthesized.append(new_record)

        items = synthesized
        annotation_ids = set(target_annotation_ids)

    # When no items survive filtering, emit diagnostics so callers can
    # understand whether this is due to annotation/participant filters or
    # LLM-positive-only score thresholds.
    if not items:
        if raw_annotation_counts:
            if llm_positive_only:
                logging.error(
                    "No classification items remained after applying "
                    "--llm-positive-only filters."
                )
                logging.error(
                    "Per-annotation candidate vs positive counts before "
                    "deduplication (annotation_id: candidates, positives):"
                )
                for ann_id in sorted(raw_annotation_counts.keys()):
                    positive = int(positive_annotation_counts.get(ann_id, 0))
                    logging.error(
                        "  %s: %d positive",
                        ann_id,
                        positive,
                    )
            else:
                logging.error(
                    "No classification items remained after applying filters; "
                    "per-annotation candidate counts before filtering:"
                )
        else:
            logging.error(
                "No classification items were found in the selected runs after "
                "applying annotation/participant filters."
            )
        return [], sorted(annotation_ids)

    # When we have surviving items and positive-only filters are active,
    # log candidate versus positive counts so annotations with zero output
    # are visible (for example, user-endorses-delusion).
    if llm_positive_only and raw_annotation_counts:
        logging.info(
            "LLM-positive candidate vs positive counts before caps and "
            "deduplication (annotation_id: candidates, positives):"
        )
        for ann_id in sorted(raw_annotation_counts.keys()):
            positive = int(positive_annotation_counts.get(ann_id, 0))
            logging.info(
                "  %s: %d positive",
                ann_id,
                positive,
            )

    # When max_items is provided, optionally resample items. For the
    # positive-only regime we support two paths:
    #   * non-randomized: enforce per-annotation caps in the streaming loop
    #     above and keep all collected items;
    #   * randomized: sample up to ``max_items`` positives per annotation id,
    #     balancing across participants using the same helper as
    #     ``classify_chats.py``.
    # For non-positive-only runs we already applied per-annotation streaming
    # caps when use_streaming_caps is True, so the shared-message sampling
    # logic is only used when max_items is zero (no cap) or when
    # llm_positive_only is enabled.
    if max_items and max_items > 0 and items and llm_positive_only:
        if not randomize:
            return items, sorted(annotation_ids)

        rng = random.Random(random_seed)
        sampled_positive_items: List[dict] = []

        # Group positive items by annotation id and participant so we can
        # allocate per-participant quotas for each code separately.
        by_annotation: Dict[str, Dict[str, List[dict]]] = {}
        for record in items:
            ann = str(record.get("annotation_id") or "").strip()
            ppt = str(record.get("participant") or "").strip()
            if not ann or not ppt:
                continue
            ann_bucket = by_annotation.setdefault(ann, {})
            ppt_bucket = ann_bucket.setdefault(ppt, [])
            ppt_bucket.append(record)

        for ann, ppt_buckets in by_annotation.items():
            ppt_names = sorted(ppt_buckets.keys())
            if not ppt_names:
                continue
            sizes = {name: len(ppt_buckets[name]) for name in ppt_names}
            total_available = sum(sizes.values())
            target = min(max_items, total_available)
            if target <= 0:
                continue

            allocations = allocate_per_participant(
                sizes,
                target,
                equal=(randomize_per_ppt == "equal"),
            )
            for name in ppt_names:
                bucket = ppt_buckets.get(name) or []
                allocated = int(allocations.get(name, 0) or 0)
                if allocated <= 0 or not bucket:
                    continue
                if allocated >= len(bucket):
                    sampled_positive_items.extend(bucket)
                else:
                    sampled_positive_items.extend(rng.sample(bucket, allocated))

        if not sampled_positive_items:
            return [], sorted(annotation_ids)

        items = sampled_positive_items
        annotation_ids = {
            str(item.get("annotation_id") or "").strip() for item in items
        }

    return items, sorted(annotation_ids)


def _gather_items_from_preprocessed(
    *,
    preprocessed_path: Path,
    transcripts_path: Path,
    annotation_id: Optional[str],
    participants: Optional[Sequence[str]],
    max_items: int,
    randomize: bool,
    random_seed: Optional[int],
    preceding_context: int,
    llm_positive_only: bool,
    llm_score_cutoff: Optional[int],
    cutoffs_by_id: Dict[str, int],
) -> Tuple[List[dict], List[str]]:
    """Collect manual-annotation items by sampling from a preprocessed table.

    This helper operates on the wide per-message annotation table produced
    by ``analysis/preprocess_annotation_family.py`` and uses the parsed
    transcripts Parquet file to recover message text. When LLM-positive-only
    filters are disabled, we sample message locations purely by index and
    then synthesise one manual item per in-scope annotation id for each
    selected message, preserving role-based scope. When LLM-positive-only
    filters are enabled, we first restrict to messages whose per-annotation
    scores exceed the configured cutoffs and then sample positives per
    annotation id and role.

    Parameters
    ----------
    preprocessed_path:
        Path to the preprocessed per-message annotations Parquet file.
    transcripts_path:
        Path to the transcripts Parquet file containing message content.
    annotation_id:
        Optional single annotation identifier to restrict the dataset. When
        omitted, all annotations present in the preprocessed table are used.
    participants:
        Optional collection of participant identifiers to restrict the
        candidate messages.
    max_items:
        Maximum number of base messages to sample per role. When zero or
        negative, all available messages for each role are used.
    randomize:
        When True, sample messages uniformly at random; otherwise, take the
        first ``max_items`` messages per role in table order.
    random_seed:
        Optional random seed for reproducible sampling.
    llm_positive_only:
        When True, restrict messages to those whose per-annotation scores
        are at or above the relevant cutoffs, sampling per annotation id.
    llm_score_cutoff:
        Optional global LLM score cutoff used as a fallback when a specific
        annotation id does not appear in ``cutoffs_by_id``.
    cutoffs_by_id:
        Mapping from annotation id to per-annotation score cutoff loaded
        from metrics or cutoffs JSON files.

    Returns
    -------
    Tuple[List[dict], List[str]]
        Tuple containing a list of manual-annotation item dictionaries and
        a sorted list of annotation identifiers included in the dataset.
    """

    resolved_pre = preprocessed_path.expanduser().resolve()
    if not resolved_pre.exists() or not resolved_pre.is_file():
        logging.error("Preprocessed annotations table not found: %s", resolved_pre)
        return [], []

    try:
        frame = pd.read_parquet(resolved_pre)
    except (OSError, ValueError) as err:
        logging.error(
            "Failed to read preprocessed annotations table %s: %s", resolved_pre, err
        )
        return [], []

    if frame.empty:
        logging.error("Preprocessed annotations table %s is empty.", resolved_pre)
        return [], []

    participant_set = {p.strip() for p in (participants or []) if p and p.strip()}
    if participant_set:
        frame = frame[frame["participant"].isin(sorted(participant_set))]
        if frame.empty:
            logging.error(
                "No messages remained after applying participant filters to %s.",
                resolved_pre,
            )
            return [], []

    # Restrict to core conversational roles that are visible in the manual
    # annotation UI.
    frame = frame[frame["role"].isin(["user", "assistant"])]
    if frame.empty:
        logging.error(
            "No user or assistant messages were found in preprocessed table %s.",
            resolved_pre,
        )
        return [], []

    # Discover available annotations from the wide score columns.
    score_columns = [name for name in frame.columns if name.startswith("score__")]
    all_annotation_ids: List[str] = sorted(
        name[len("score__") :] for name in score_columns if len(name) > len("score__")
    )
    if annotation_id:
        target_annotation_ids = (
            [annotation_id] if annotation_id in all_annotation_ids else []
        )
    else:
        target_annotation_ids = list(all_annotation_ids)
    if not target_annotation_ids:
        logging.error(
            "No usable annotations discovered in %s (requested annotation=%s).",
            resolved_pre,
            annotation_id or "<none>",
        )
        return [], []

    # Compute role-scoped annotation metadata so that user-only and
    # assistant-only codes are only paired with compatible messages.
    scope_by_annotation_id: Dict[str, Optional[Set[str]]] = {}
    for ann_id in target_annotation_ids:
        try:
            spec = resolve_annotation(ann_id)
        except ValueError:
            scope_by_annotation_id[ann_id] = None
            continue
        allowed_roles = parse_annotation_scope(spec)
        scope_by_annotation_id[ann_id] = allowed_roles

    # Clamp preceding context to a reasonable upper bound consistent with
    # the CLI help text.
    if preceding_context < 0:
        preceding = 0
    elif preceding_context > 10:
        preceding = 10
    else:
        preceding = preceding_context

    rng: Optional[random.Random] = None
    if randomize:
        rng = random.Random(random_seed)

    # Recover transcript metadata so we can attach text and timestamps.
    resolved_tx = transcripts_path.expanduser().resolve()
    if not resolved_tx.exists() or not resolved_tx.is_file():
        logging.error("Transcripts table not found: %s", resolved_tx)
        return [], []

    items: List[dict] = []

    if llm_positive_only:
        # Positive-only regime: restrict to messages whose per-annotation
        # scores exceed the configured cutoffs and sample per annotation id
        # and role.
        if not cutoffs_by_id and llm_score_cutoff is None:
            logging.error(
                "Requested --llm-positive-only with --preprocessed-table but "
                "no per-annotation or global LLM score cutoffs were provided."
            )
            return [], []

        output_annotation_ids: Set[str] = set()
        selected_rows: List[Dict[str, object]] = []

        # First phase: restrict to rows whose scores meet the cutoffs and
        # sample up to ``max_items`` indices per annotation id and role.
        for ann_id in target_annotation_ids:
            score_column = f"score__{ann_id}"
            if score_column not in frame.columns:
                continue

            cutoff_for_annotation = cutoffs_by_id.get(ann_id, llm_score_cutoff)
            if cutoff_for_annotation is None:
                logging.warning(
                    "No LLM score cutoff available for annotation %s; "
                    "skipping positives for this code in preprocessed mode.",
                    ann_id,
                )
                continue

            ann_frame = frame[
                frame[score_column].notna()
                & (frame[score_column] >= cutoff_for_annotation)
            ]
            if ann_frame.empty:
                continue

            for role_name in ("user", "assistant"):
                role_frame = ann_frame[ann_frame["role"] == role_name]
                if role_frame.empty:
                    continue
                if max_items and max_items > 0:
                    n_target = min(max_items, len(role_frame))
                else:
                    n_target = len(role_frame)
                if n_target <= 0:
                    continue
                if randomize and rng is not None:
                    indices = list(role_frame.index)
                    selected_indices = rng.sample(indices, n_target)
                    role_selected = role_frame.loc[selected_indices]
                else:
                    role_selected = role_frame.iloc[:n_target]

                for _, row in role_selected.iterrows():
                    participant = str(row.get("participant") or "").strip()
                    source_path = str(row.get("source_path") or "").strip()
                    chat_index = int(row.get("chat_index"))
                    message_index = int(row.get("message_index"))

                    score_value = row.get(score_column)
                    try:
                        llm_score_int: Optional[int] = (
                            int(round(float(score_value)))
                            if score_value is not None
                            else None
                        )
                    except (TypeError, ValueError, OverflowError):
                        llm_score_int = None
                    llm_positive_flag: Optional[bool] = None
                    if llm_score_int is not None and cutoff_for_annotation is not None:
                        llm_positive_flag = llm_score_int >= int(cutoff_for_annotation)

                    selected_rows.append(
                        {
                            "participant": participant,
                            "source_path": source_path,
                            "chat_index": chat_index,
                            "message_index": message_index,
                            "annotation_id": ann_id,
                            "llm_score": llm_score_int,
                            "llm_positive": llm_positive_flag,
                        }
                    )

        if not selected_rows:
            logging.error(
                "No positive manual-annotation items were selected from "
                "preprocessed table %s before joining transcripts %s.",
                resolved_pre,
                resolved_tx,
            )
            return [], sorted(target_annotation_ids)

        # Second phase: join the sampled indices against the transcripts
        # table in bulk and then build preceding context per chat. This
        # avoids scanning the full transcripts table for every item.
        transcripts_frame = _load_transcripts_table(resolved_tx)
        if transcripts_frame is None or transcripts_frame.empty:
            logging.error(
                "Transcripts table %s is unavailable when building "
                "positive-only preprocessed items.",
                resolved_tx,
            )
            return [], sorted(target_annotation_ids)

        selected_df = pd.DataFrame(selected_rows)
        if selected_df.empty:
            logging.error(
                "No positive manual-annotation items were selected from "
                "preprocessed table %s before joining transcripts %s.",
                resolved_pre,
                resolved_tx,
            )
            return [], sorted(target_annotation_ids)

        selected_df = selected_df.drop_duplicates(
            subset=[
                "participant",
                "source_path",
                "chat_index",
                "message_index",
                "annotation_id",
            ]
        )

        participant_values = [
            str(value) for value in selected_df["participant"].unique() if value
        ]
        source_values = [
            str(value) for value in selected_df["source_path"].unique() if value
        ]
        chat_values = [int(value) for value in selected_df["chat_index"].unique()]

        transcripts_subset = transcripts_frame[
            transcripts_frame["participant"].isin(participant_values)
            & transcripts_frame["source_path"].isin(source_values)
            & transcripts_frame["chat_index"].isin(chat_values)
        ]
        if transcripts_subset.empty:
            logging.error(
                "No transcript rows were found for positive preprocessed items "
                "in %s.",
                resolved_tx,
            )
            return [], sorted(target_annotation_ids)

        chats_by_key: Dict[Tuple[str, str, int], pd.DataFrame] = {}
        for (ppt, src, chat_idx), group in transcripts_subset.groupby(
            ["participant", "source_path", "chat_index"]
        ):
            chats_by_key[(str(ppt), str(src), int(chat_idx))] = group.sort_values(
                "message_index"
            )

        items = []
        for _, row in selected_df.iterrows():
            participant = str(row.get("participant") or "").strip()
            source_path = str(row.get("source_path") or "").strip()
            chat_index = int(row.get("chat_index"))
            message_index = int(row.get("message_index"))
            ann_id = str(row.get("annotation_id") or "").strip()

            chat_key = (participant, source_path, chat_index)
            chat_frame = chats_by_key.get(chat_key)
            if chat_frame is None or chat_frame.empty:
                logging.warning(
                    "No transcript chat found in %s for %s/%s (%d).",
                    resolved_tx,
                    participant,
                    source_path,
                    chat_index,
                )
                continue

            llm_score_value = row.get("llm_score")
            try:
                llm_score_int = (
                    int(llm_score_value) if llm_score_value is not None else None
                )
            except (TypeError, ValueError, OverflowError):
                llm_score_int = None

            llm_positive_flag_value = row.get("llm_positive")
            llm_positive_flag: Optional[bool] = None
            if llm_positive_flag_value is not None:
                llm_positive_flag = bool(llm_positive_flag_value)

            allowed_roles = scope_by_annotation_id.get(ann_id)
            record = _build_item_from_chat_frame(
                chat_frame,
                participant=participant,
                source_path=source_path,
                chat_index=chat_index,
                message_index=message_index,
                annotation_id=ann_id,
                allowed_roles=allowed_roles,
                preceding_context=preceding,
                llm_score=llm_score_int,
                llm_positive=llm_positive_flag,
                transcripts_path=resolved_tx,
            )
            if record is None:
                continue

            items.append(record)
            if ann_id:
                output_annotation_ids.add(ann_id)

        if not items:
            logging.error(
                "No positive manual-annotation items were synthesised from "
                "preprocessed table %s and transcripts %s.",
                resolved_pre,
                resolved_tx,
            )
            return [], sorted(target_annotation_ids)

        return items, sorted(output_annotation_ids or target_annotation_ids)

    # Non-positive regime: sample up to ``max_items`` base messages per role
    # by index and synthesise one item per in-scope annotation id for each
    # selected message.
    sampled_rows: List[Dict[str, object]] = []
    for role_name in ("user", "assistant"):
        role_frame = frame[frame["role"] == role_name]
        if role_frame.empty:
            continue
        if max_items and max_items > 0:
            n_target = min(max_items, len(role_frame))
        else:
            n_target = len(role_frame)
        if n_target <= 0:
            continue
        if randomize and rng is not None:
            indices = list(role_frame.index)
            selected_indices = rng.sample(indices, n_target)
            role_selected = role_frame.loc[selected_indices]
        else:
            role_selected = role_frame.iloc[:n_target]
        for _, row in role_selected.iterrows():
            sampled_rows.append(
                {
                    "participant": str(row.get("participant") or "").strip(),
                    "source_path": str(row.get("source_path") or "").strip(),
                    "chat_index": int(row.get("chat_index")),
                    "message_index": int(row.get("message_index")),
                    "role": str(row.get("role") or "").strip(),
                }
            )

    if not sampled_rows:
        logging.error(
            "No base messages were selected from preprocessed table %s "
            "after applying filters and sampling.",
            resolved_pre,
        )
        return [], []

    for base in sampled_rows:
        participant = str(base["participant"])
        source_path = str(base["source_path"])
        chat_index = int(base["chat_index"])
        message_index = int(base["message_index"])

        for ann_id in target_annotation_ids:
            allowed_roles = scope_by_annotation_id.get(ann_id)
            record = _build_preprocessed_item(
                resolved_tx,
                participant=participant,
                source_path=source_path,
                chat_index=chat_index,
                message_index=message_index,
                annotation_id=ann_id,
                allowed_roles=allowed_roles,
                preceding_context=preceding,
            )
            if record is None:
                continue
            items.append(record)

    if not items:
        logging.error(
            "No manual-annotation items were synthesised from preprocessed "
            "table %s and transcripts %s.",
            resolved_pre,
            resolved_tx,
        )
        return [], sorted(target_annotation_ids)

    return items, sorted(target_annotation_ids)


def _write_dataset(
    path: Path,
    *,
    items: Sequence[dict],
    annotation_ids: Sequence[str],
    preceding_context: int,
    classification_root: Optional[str] = None,
    classification_runs: Optional[Sequence[str]] = None,
) -> int:
    """Write a JSONL manual dataset for the given items."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logging.error("Failed to create output directory %s: %s", path.parent, err)
        return 0

    transcripts_root = get_default_transcripts_root()
    meta_record: Dict[str, object] = {
        "type": "meta",
        "generated_at": datetime.now().isoformat(),
        "input_root": str(transcripts_root),
        "annotation_ids": list(annotation_ids),
        "annotation_snapshots": {},
        "preceding_context": int(preceding_context),
        "classification_root": classification_root,
        "classification_runs": list(classification_runs or []),
        "parameters": {
            "participants": sorted(
                {
                    str(item.get("participant") or "")
                    for item in items
                    if item.get("participant")
                }
            ),
            "max_messages": len(items),
            "max_conversations": None,
            "randomize": False,
            "randomize_per_ppt": "proportional",
            "randomize_conversations": False,
            "reverse_conversations": False,
            "follow_links": False,
        },
    }

    count = 0
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
            seq = 0
            for item in items:
                record = dict(item)
                record["type"] = "item"
                record["sequence_index"] = seq
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                seq += 1
                count += 1
    except OSError as err:
        logging.error("Failed to write dataset to %s: %s", path, err)
        return 0

    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if (
        args.llm_positive_only
        and not args.llm_cutoffs_json
        and args.llm_score_cutoff is None
    ):
        logging.warning(
            "Requested --llm-positive-only without --llm-cutoffs-json or "
            "--llm-score-cutoff; no score-based filtering will be applied."
        )

    # When a metrics or cutoffs JSON is provided, load per-annotation score
    # thresholds so that LLM-positive filtering can respect the canonical
    # operating points used elsewhere in the analysis pipeline.
    cutoffs_by_id: Dict[str, int] = {}
    if getattr(args, "llm_cutoffs_json", None):
        mapping = load_llm_cutoffs_from_json(args.llm_cutoffs_json)
        if mapping is None:
            logging.error(
                "Failed to load per-annotation LLM cutoffs from %s.",
                args.llm_cutoffs_json,
            )
            return 2
        cutoffs_by_id = dict(mapping)

    # Sample messages directly from the preprocessed Parquet table using
    # index-based selection. This path bypasses JSONL classification files
    # entirely and relies on transcripts_data for content.
    items, annotation_ids = _gather_items_from_preprocessed(
        preprocessed_path=args.preprocessed_table,
        transcripts_path=args.transcripts_table,
        annotation_id=args.annotation_id,
        participants=args.participants,
        max_items=args.max_items,
        randomize=bool(getattr(args, "randomize", False)),
        random_seed=getattr(args, "random_seed", None),
        preceding_context=int(getattr(args, "preceding_context", 0) or 0),
        llm_positive_only=bool(args.llm_positive_only),
        llm_score_cutoff=args.llm_score_cutoff,
        cutoffs_by_id=cutoffs_by_id,
    )
    if not items:
        logging.error("No classification items were collected for the dataset.")
        return 2

    # Enforce a hard per-annotation cap so that the final manual-annotation
    # dataset never contains more than ``max_items`` messages for any single
    # annotation id, regardless of how many roles or message locations were
    # used during sampling.
    if args.max_items and args.max_items > 0:
        items = _cap_items_per_annotation(
            items,
            max_items=args.max_items,
            randomize=bool(getattr(args, "randomize", False)),
            random_seed=getattr(args, "random_seed", None),
        )

    participants_in_items = sorted(
        {
            str(item.get("participant") or "")
            for item in items
            if item.get("participant")
        }
    )
    logging.info(
        "Discovered %d unique participants in manual dataset: %s",
        len(participants_in_items),
        ", ".join(participants_in_items) if participants_in_items else "<none>",
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = _build_output_path(
            model=args.model, annotation_id=args.annotation_id, args=args
        )

    written = _write_dataset(
        output_path,
        items=items,
        annotation_ids=annotation_ids,
        preceding_context=args.preceding_context,
        classification_root=None,
        classification_runs=None,
    )
    if written <= 0:
        logging.error("No records were written to %s.", output_path)
        return 2

    logging.info(
        "Wrote %s manual annotation items for model %s to %s.",
        written,
        args.model,
        output_path,
    )

    # Summarise the final item counts per annotation id so that callers can
    # quickly verify how many messages were included for each code.
    counts: Counter[str] = Counter()
    for item in items:
        ann = str(item.get("annotation_id") or "").strip()
        if not ann:
            continue
        counts[ann] += 1
    if counts:
        sys.stdout.write("\nManual-annotation item counts per annotation id:\n")
        sys.stdout.write(f"{'annotation_id':<40} {'n_items':>8}\n")
        for ann_id in sorted(counts.keys()):
            sys.stdout.write(f"{ann_id:<40} {counts[ann_id]:8d}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
