"""Utility to extract message subsets from de-identified transcripts.

Reads a CSV (subsets.csv) specifying which transcript file, conversation (optional),
and quote to target. For each row, finds the matching message and writes a flat list
of messages around it (N before, M after) to `subsets/` while mirroring the directory
structure under `transcripts_de_ided/`.

CSV columns (header required):
  - rel_path: Path relative to `--input-dir` (default: transcripts_de_ided)
  - conversation_id: Conversation title/identifier, or blank if not applicable
  - quote: Substring to locate within message content
  - label: one of normal | pivotal | harmful
  - participant: Participant identifier (ppt)
  - prev_count: How many messages before the matched message to include
  - after_count: How many messages after the matched message to include

Command line:
  python scripts/make_subsets.py \
    --csv subsets.csv \
    --input-dir transcripts_de_ided \
    --output-dir subsets \
    [--dry-run] [--verbose]

Returns exit code 0 on success; nonzero on any failures.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from chat import (
    Chat,
    find_message_index_by_quote,
    load_chats_for_file,
    select_chat_by_title_or_quote,
)
from utils.io import get_default_transcripts_root
from utils.schema import (
    PLAN_REQUIRED_COLUMNS,
    SUBSET_INFO_COMMENTS,
    SUBSET_INFO_CONVERSATION_ID,
    SUBSET_INFO_CONVERSATION_TITLE,
    SUBSET_INFO_GENERATED_UTC,
    SUBSET_INFO_KEY,
    SUBSET_INFO_LABEL,
    SUBSET_INFO_MATCH_INDEX,
    SUBSET_INFO_PARTICIPANT,
    SUBSET_INFO_RANGE_INCLUSIVE,
    SUBSET_INFO_ROW,
    SUBSET_INFO_SOURCE_REL_PATH,
    SUBSET_INFO_SOURCE_TOTAL_MESSAGES,
    SUBSET_MESSAGES_KEY,
)
from utils.utils import ensure_dir, pick_title_string, resolve_source_path, short_slug


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters:
        argv: Optional sequence of raw arguments. Defaults to `sys.argv[1:]`.

    Returns:
        Parsed `argparse.Namespace` with fields used by `main`.
    """

    p = argparse.ArgumentParser(
        description="Extract subsets from de-identified transcripts"
    )

    p.add_argument("--csv", default="subsets.csv", help="Path to CSV instructions file")
    p.add_argument(
        "--input-dir",
        default=str(get_default_transcripts_root()),
        help="Root directory of de-identified transcripts",
    )
    p.add_argument(
        "--output-dir",
        default="subsets",
        help="Output root directory to mirror input structure",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument(
        "--prev-count",
        type=int,
        default=None,
        help="Override CSV prev_count for all rows (messages before match)",
    )
    p.add_argument(
        "--after-count",
        type=int,
        default=None,
        help="Override CSV after_count for all rows (messages after match)",
    )
    p.add_argument(
        "--annotations-parquet",
        type=str,
        default="annotations/all_annotations__preprocessed.parquet",
        help=(
            "Optional path to the canonical per-message annotations table. "
            "When present, per-message annotation scores are attached to each "
            "subset message using the 'annotation_scores' field."
        ),
    )
    return p.parse_args(argv)


def _coerce_int(val: str, field: str) -> int:
    """Convert a CSV string field to int with clear error messages.

    Parameters:
        val: Raw string value from CSV.
        field: Field name for error reporting.

    Returns:
        Parsed integer.
    """

    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {field!r}: {val!r}") from exc


# Generic selection and small helpers live in src/chat and scripts/utils.


@dataclass
class OutputNameContext:
    """Context for naming an output subset file."""

    output_root: str
    rel_path: str
    rowno: int
    conv_title: Optional[str]
    data: Dict[str, Any]
    quote_text: str
    label: str


def _build_output_path(ctx: OutputNameContext) -> str:
    """Compute the output JSON file path for a subset selection.

    Mirrors the `rel_path` under `output_root` and names the file as:
      `{label}_{short-title}_{short-quote}.json`

    Parameters:
        output_root: Root output directory.
        rel_path: Relative path under input root to the source JSON file.
        rowno: 1-based CSV row number (for disambiguation if needed).
        conv_title: Optional conversation title.
        data: Parsed source JSON (used to derive a title when needed).
        quote_text: The matched quote to summarize in the filename.
        label: normal | pivotal | harmful.

    Returns:
        Destination file path.
    """

    rel_path = ctx.rel_path.strip().lstrip("/\\")
    out_dir = os.path.join(ctx.output_root, os.path.dirname(rel_path))
    ensure_dir(out_dir)

    title_str = pick_title_string(ctx.data, ctx.conv_title, rel_path)
    title_slug = short_slug(title_str)
    quote_slug = short_slug(ctx.quote_text)
    base_name = f"{ctx.label}_{title_slug}_{quote_slug}.json"
    dest_path = os.path.join(out_dir, base_name)
    return dest_path


def _write_subset(dest_path: str, payload: Dict[str, Any], dry_run: bool) -> None:
    """Write the subset JSON payload to disk.

    Parameters:
        dest_path: Output file path.
        payload: Full JSON dictionary to write.
        dry_run: If True, do not write the file.
    """

    if dry_run:
        return

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _validate_label(value: str) -> str:
    """Validate the label column and normalize it to lowercase.

    Parameters:
        value: Raw string label from CSV.

    Returns:
        Normalized label string.
    """

    allowed = {"normal", "pivotal", "harmful"}
    val = value.strip().lower()
    if val not in allowed:
        raise ValueError(f"label must be one of {sorted(allowed)}; got {value!r}")
    return val


@dataclass
class RunConfig:
    """Execution parameters for processing a row."""

    input_dir: str
    output_dir: str
    dry_run: bool
    verbose: bool
    annotations_index: Dict[Tuple[str, str, int, int], Dict[str, Any]]
    override_prev: Optional[int] = None
    override_after: Optional[int] = None


@dataclass
class RowParams:
    """Parsed CSV fields for one row.

    The `window` field is a (prev, after) tuple controlling the subset size.
    """

    rel_path: str
    conversation_id: Optional[str]
    quote: str
    label: str
    participant: str
    window: Tuple[int, int]
    comments: Optional[str]


@dataclass
class SubsetExtract:
    """Selected slice of messages and related indices."""

    conv_title: Optional[str]
    chat_index: int
    match_index: int
    start: int
    end: int
    total_messages: int
    messages: List[Dict[str, Any]]


def _build_subset(
    chat: Chat,
    params: RowParams,
    match_index: int,
    *,
    chat_index: int,
) -> SubsetExtract:
    """Build a SubsetExtract window around ``match_index`` for a chat."""

    msgs = chat.messages
    start_i = max(0, match_index - params.window[0])
    end_i = min(len(msgs) - 1, match_index + params.window[1])
    subset = [msgs[i] for i in range(start_i, end_i + 1)]
    return SubsetExtract(
        conv_title=chat.key,
        chat_index=chat_index,
        match_index=match_index,
        start=start_i,
        end=end_i,
        total_messages=len(msgs),
        messages=subset,
    )


def _find_best_fuzzy_match(
    all_chats: List[Chat], quote_text: str
) -> Optional[Tuple[Chat, int, float]]:
    """Return the closest fuzzy match for ``quote_text`` across all messages."""

    needle = quote_text.strip()
    if not needle:
        return None

    best_chat: Optional[Chat] = None
    best_index: Optional[int] = None
    best_score = 0.0
    needle_lower = needle.lower()

    for candidate_chat in all_chats:
        for idx_msg, msg in enumerate(candidate_chat.messages):
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            score = SequenceMatcher(None, needle_lower, content.lower()).ratio()
            if score > best_score:
                best_score = score
                best_chat = candidate_chat
                best_index = idx_msg

    if best_chat is None or best_index is None:
        return None
    return best_chat, best_index, best_score


def _extract_subset_with_fallback(
    src: Path,
    initial_chats: List[Chat],
    params: RowParams,
    *,
    chat_index_by_key: Dict[str, int],
) -> SubsetExtract:
    """Extract a subset, searching all branches and suggesting a nearby match."""

    chat = select_chat_by_title_or_quote(
        initial_chats,
        title=params.conversation_id,
        quote=params.quote,
    )
    chat_index = chat_index_by_key.get(chat.key, 0)
    try:
        idx_match = find_message_index_by_quote(chat.messages, params.quote)
        return _build_subset(
            chat,
            params,
            idx_match,
            chat_index=chat_index,
        )
    except ValueError as exc:
        message = str(exc)
        if "Quote not found in any message content" not in message:
            raise

    # Fallback: search across all visible messages from the ChatGPT mapping.
    chats_all = load_chats_for_file(src, strategy="all_messages")
    if not chats_all:
        preview = params.quote
        max_preview = 80
        if len(preview) > max_preview:
            preview = preview[: max_preview - 3] + "..."
        raise ValueError(
            "Quote not found in main conversation path and no conversations "
            "available when searching full mapping; "
            f"preview={preview!r}, length={len(params.quote)}, "
            f"messages_main={len(chat.messages)}"
        )

    exact_matches: List[Tuple[Chat, int]] = []
    for candidate_chat in chats_all:
        try:
            idx_match = find_message_index_by_quote(
                candidate_chat.messages, params.quote
            )
        except ValueError:
            continue
        exact_matches.append((candidate_chat, idx_match))

    if len(exact_matches) == 1:
        match_chat, match_idx = exact_matches[0]
        return _build_subset(
            match_chat,
            params,
            match_idx,
            chat_index=chat_index_by_key.get(match_chat.key, 0),
        )

    if len(exact_matches) > 1:
        if params.conversation_id:
            filtered = [
                (c_chat, m_idx)
                for (c_chat, m_idx) in exact_matches
                if c_chat.key.lower() == params.conversation_id.lower()
            ]
            if len(filtered) == 1:
                match_chat, match_idx = filtered[0]
                return _build_subset(
                    match_chat,
                    params,
                    match_idx,
                    chat_index=chat_index_by_key.get(match_chat.key, 0),
                )

        conv_keys = sorted({c_chat.key for (c_chat, _idx_match) in exact_matches})
        preview = params.quote
        max_preview = 80
        if len(preview) > max_preview:
            preview = preview[: max_preview - 3] + "..."
        raise ValueError(
            "Quote appears in multiple conversations when searching full "
            "mapping; refine conversation_id or quote; "
            f"preview={preview!r}, length={len(params.quote)}, "
            f"candidates={conv_keys}"
        )

    # No exact matches even when searching all visible messages; suggest
    # the closest nearby message for manual review.
    suggestion = _find_best_fuzzy_match(chats_all, params.quote)
    preview = params.quote
    max_preview = 80
    if len(preview) > max_preview:
        preview = preview[: max_preview - 3] + "..."

    if suggestion is None:
        raise ValueError(
            "Quote not found in any message content even after searching "
            "full mapping; "
            f"preview={preview!r}, length={len(params.quote)}, "
            f"messages_main={len(chat.messages)}, "
            f"messages_full={sum(len(c.messages) for c in chats_all)}"
        )

    sug_chat, sug_index, score = suggestion
    sug_msg = sug_chat.messages[sug_index]
    sug_content = sug_msg.get("content", "")
    sug_preview = (
        sug_content
        if isinstance(sug_content, str) and len(sug_content) <= max_preview
        else str(sug_content)[: max_preview - 3] + "..."
    )
    raise ValueError(
        "Quote not found in any message content even after searching full "
        "mapping; "
        f"preview={preview!r}, length={len(params.quote)}, "
        f"messages_main={len(chat.messages)}, "
        f"messages_full={sum(len(c.messages) for c in chats_all)}, "
        f"closest_match_conversation={sug_chat.key!r}, "
        f"closest_match_index={sug_index}, "
        f"closest_match_similarity={score:.3f}, "
        f"closest_match_preview={sug_preview!r}"
    )


def _load_annotations_index(
    annotations_parquet: Optional[str],
) -> Dict[Tuple[str, str, int, int], Dict[str, Any]]:
    """Return an index mapping message keys to annotation scores.

    The index key has the shape ``(participant, source_path, chat_index,
    message_index)`` and values are dictionaries containing only
    ``score__*`` columns from the preprocessed annotations table.
    """

    if not annotations_parquet:
        return {}

    path = Path(annotations_parquet).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}

    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return {}

    required_cols = {"participant", "source_path", "chat_index", "message_index"}
    if not required_cols.issubset(set(frame.columns)):
        return {}

    score_cols = [col for col in frame.columns if str(col).startswith("score__")]
    if not score_cols:
        return {}

    subset = frame[list(required_cols) + score_cols].copy()

    index: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    for _row_index, row in subset.iterrows():
        participant = str(row.get("participant", "")).strip()
        source_path = str(row.get("source_path", "")).strip()
        try:
            chat_index = int(row.get("chat_index"))
            message_index = int(row.get("message_index"))
        except (TypeError, ValueError):
            continue

        key = (participant, source_path, chat_index, message_index)
        scores: Dict[str, Any] = {}
        for col in score_cols:
            value = row.get(col)
            # Treat NaN/None as missing.
            if value is None:
                continue
            try:
                # Many score columns are numeric; coerce where reasonable.
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = value
            scores[str(col)] = numeric_value
        if not scores:
            continue
        index[key] = scores
    return index


def _process_row(idx: int, row: Dict[str, str], cfg: RunConfig) -> Optional[str]:
    """Process a single CSV row.

    Parameters:
        idx: 1-based row index (excluding header).
        row: Dict of CSV columns for this row.
        input_dir: Root directory of transcripts.
        output_dir: Root directory for subsets.
        dry_run: If True, do not write outputs.
        verbose: If True, print progress to stderr.

    Returns:
        None on success; error message string on failure.
    """

    params = RowParams(
        rel_path=(row.get("rel_path") or "").strip(),
        conversation_id=(row.get("conversation_id") or "").strip() or None,
        quote=(row.get("quote") or "").strip(),
        label=_validate_label((row.get("label") or "").strip()),
        participant=(row.get("participant") or "").strip(),
        window=(
            _coerce_int((row.get("prev_count") or "0"), "prev_count"),
            _coerce_int((row.get("after_count") or "0"), "after_count"),
        ),
        comments=(row.get("comments") or "").strip() or None,
    )

    # Apply global overrides if provided
    if cfg.override_prev is not None or cfg.override_after is not None:
        prev = cfg.override_prev if cfg.override_prev is not None else params.window[0]
        after = (
            cfg.override_after if cfg.override_after is not None else params.window[1]
        )
        params = RowParams(
            rel_path=params.rel_path,
            conversation_id=params.conversation_id,
            quote=params.quote,
            label=params.label,
            participant=params.participant,
            window=(prev, after),
            comments=params.comments,
        )

    if not params.rel_path:
        return "Empty rel_path"
    if not params.quote:
        return "Empty quote"

    try:
        src = resolve_source_path(cfg.input_dir, params.rel_path)
        src_path = Path(src)
        chats = load_chats_for_file(src_path)
        chat_index_by_key = {chat.key: chat_idx for chat_idx, chat in enumerate(chats)}
        extracted = _extract_subset_with_fallback(
            src_path,
            chats,
            params,
            chat_index_by_key=chat_index_by_key,
        )
        meta = None

        dest = _build_output_path(
            OutputNameContext(
                output_root=cfg.output_dir,
                rel_path=params.rel_path,
                rowno=idx,
                conv_title=extracted.conv_title,
                data={},
                quote_text=params.quote,
                label=params.label,
            )
        )

        annotations_index = cfg.annotations_index
        messages_with_scores: List[Dict[str, Any]] = []
        for message_index in range(extracted.start, extracted.end + 1):
            local_index = message_index - extracted.start
            raw_msg = extracted.messages[local_index]
            message = dict(raw_msg)
            if params.participant and annotations_index:
                key = (
                    params.participant,
                    params.rel_path,
                    extracted.chat_index,
                    message_index,
                )
                scores = annotations_index.get(key)
                if scores:
                    message["annotation_scores"] = scores
            messages_with_scores.append(message)

        payload: Dict[str, Any] = {
            "autogenerated_notice": (
                "AUTO-GENERATED FILE. Edit with care: changes may be "
                "overwritten if this subset is regenerated."
            ),
            SUBSET_INFO_KEY: {
                SUBSET_INFO_ROW: idx,
                SUBSET_INFO_LABEL: params.label,
                SUBSET_INFO_PARTICIPANT: params.participant,
                SUBSET_INFO_SOURCE_REL_PATH: params.rel_path,
                SUBSET_INFO_CONVERSATION_ID: params.conversation_id,
                SUBSET_INFO_CONVERSATION_TITLE: extracted.conv_title,
                SUBSET_INFO_MATCH_INDEX: extracted.match_index,
                SUBSET_INFO_RANGE_INCLUSIVE: [extracted.start, extracted.end],
                SUBSET_INFO_SOURCE_TOTAL_MESSAGES: extracted.total_messages,
                SUBSET_INFO_GENERATED_UTC: _dt.datetime.now(
                    _dt.timezone.utc
                ).isoformat(),
            },
            SUBSET_MESSAGES_KEY: messages_with_scores,
        }
        if params.comments:
            payload[SUBSET_INFO_KEY][SUBSET_INFO_COMMENTS] = params.comments
        if meta:
            payload["meta"] = meta

        _write_subset(dest_path=dest, payload=payload, dry_run=cfg.dry_run)
        if cfg.verbose:
            print(
                f"Row {idx}: wrote {dest} " f"[{extracted.start}-{extracted.end}]",
                file=sys.stderr,
            )
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if cfg.verbose:
            print(f"Row {idx} FAILED: {exc}", file=sys.stderr)
        return str(exc)


def process_csv(
    csv_path: str,
    input_dir: str,
    output_dir: str,
    dry_run: bool,
    verbose: bool,
    *,
    override_prev: Optional[int] = None,
    override_after: Optional[int] = None,
    annotations_parquet: Optional[str] = None,
) -> int:
    """Process all rows in the CSV and write subset JSON files.

    Parameters:
        csv_path: Path to the CSV file to read.
        input_dir: Root directory containing transcript JSON files.
        output_dir: Root directory to write subset files into.
        dry_run: If True, compute and log but do not write outputs.
        verbose: If True, prints per-row progress.

    Returns:
        0 on success; non-zero if any row fails (after attempting all rows).
    """

    failures: List[Tuple[int, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = PLAN_REQUIRED_COLUMNS
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        annotations_index = _load_annotations_index(annotations_parquet)

        cfg = RunConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            dry_run=dry_run,
            verbose=verbose,
            annotations_index=annotations_index,
            override_prev=override_prev,
            override_after=override_after,
        )
        for idx, row in enumerate(reader, start=1):
            err = _process_row(idx=idx, row=row, cfg=cfg)
            if err:
                failures.append((idx, err))

    if not failures:
        return 0

    if not verbose:
        for rowno, err in failures:
            print(f"Error on row {rowno}: {err}", file=sys.stderr)
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point.

    Parameters:
        argv: Optional sequence of command-line args.

    Returns:
        Process exit code.
    """

    args = _parse_args(argv)
    return process_csv(
        csv_path=args.csv,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        override_prev=args.prev_count,
        override_after=args.after_count,
        annotations_parquet=args.annotations_parquet,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
