"""Build a subset plan CSV from annotation conversation counts.

This utility connects annotation-level conversation aggregates to the subset
generation pipeline. It reads a single conversation-counts CSV produced by
``scripts/annotation/annotation_conversation_counts.py`` and reconstructs a
plan CSV compatible with ``make_subsets.py``.

For each conversation row that passed the annotation filters, the script
discovers positive annotation records in the underlying JSONL outputs and
emits one subset row per conversation. Each row specifies:

- ``rel_path``: Transcript path relative to ``transcripts_de_ided/``.
- ``conversation_id``: Conversation key/title when available.
- ``quote``: A short snippet from one positive message.
- ``label``: Subset label (default: ``harmful``).
- ``participant``: Participant identifier.
- ``prev_count`` / ``after_count``: Window size around the quote.

The generated CSV is intended as an automatic starting point that can be
manually pruned and refined before running ``make_subsets.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from annotation.annotation_prompts import should_count_positive
from annotation.conversation_counts import ConversationCountsRow
from annotation.io import extract_conversation_key, iter_annotation_records
from utils.cli import add_annotation_outputs_arguments, add_score_cutoff_argument
from utils.io import collect_family_files, write_dicts_to_csv
from utils.param_strings import string_to_dict
from utils.schema import (
    PLAN_COLUMN_AFTER_COUNT,
    PLAN_COLUMN_CONVERSATION_ID,
    PLAN_COLUMN_LABEL,
    PLAN_COLUMN_PARTICIPANT,
    PLAN_COLUMN_PREV_COUNT,
    PLAN_COLUMN_QUOTE,
    PLAN_COLUMN_REL_PATH,
)

ConversationKey = tuple[str, str, int]


@dataclass
class PositiveInstance:
    """Single positive annotation instance within a conversation."""

    message_index: int
    content: str


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the plan builder.

    Parameters
    ----------
    argv:
        Optional raw argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including the reference annotation JSONL path,
        outputs root, target annotation id, score cutoff, counts CSV, and
        output plan CSV location.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build a subsets plan CSV from conversation-level annotation counts."
        )
    )
    add_annotation_outputs_arguments(
        parser,
        file_help=(
            "Reference JSONL file from the annotation run. All sibling files "
            "with the same basename under the outputs root will be scanned "
            "when reconstructing positive messages."
        ),
    )
    parser.add_argument(
        "--conversation-counts-csv",
        type=Path,
        required=True,
        help=(
            "Conversation counts CSV produced by "
            "scripts/annotation/annotation_conversation_counts.py."
        ),
    )
    parser.add_argument(
        "--annotation-id",
        "-a",
        required=True,
        help="Annotation identifier used when generating the counts CSV.",
    )
    add_score_cutoff_argument(parser)
    parser.add_argument(
        "--label",
        "-l",
        default="harmful",
        help=(
            "Subset label to assign in the plan CSV (default: harmful). "
            "Common values are normal, pivotal, and harmful."
        ),
    )
    parser.add_argument(
        "--prev-count",
        type=int,
        default=3,
        help=(
            "Number of messages before the matched quote to include in each "
            "subset window (default: 3)."
        ),
    )
    parser.add_argument(
        "--after-count",
        type=int,
        default=3,
        help=(
            "Number of messages after the matched quote to include in each "
            "subset window (default: 3)."
        ),
    )
    parser.add_argument(
        "--output-plan-csv",
        type=Path,
        default=None,
        help=(
            "Optional output path for the generated plan CSV. When omitted, "
            "a file is written under subsets/auto_subsets/ with the same "
            "basename as the conversation-counts CSV."
        ),
    )
    return parser.parse_args(argv)


def _build_default_plan_path(conversation_counts_csv: Path) -> Path:
    """Return a default plan CSV path derived from the counts CSV name."""

    base_dir = Path("subsets") / "auto_subsets"
    base_dir = base_dir.expanduser()
    return base_dir / conversation_counts_csv.name


def _load_conversation_counts(
    csv_path: Path,
) -> dict[ConversationKey, ConversationCountsRow]:
    """Return conversation-level stats indexed by conversation key."""

    mapping: dict[ConversationKey, ConversationCountsRow] = {}
    csv_path = csv_path.expanduser().resolve()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            participant = (row.get("participant") or "").strip()
            transcript_rel_path = (row.get("transcript_rel_path") or "").strip()
            conv_index_raw = row.get("conversation_index")
            conv_key_raw = row.get("conversation_key")
            date_value = row.get("conversation_date")
            pos_raw = row.get("positive_count")
            total_raw = row.get("total_messages_in_run_for_conv")
            if not participant or not transcript_rel_path or conv_index_raw is None:
                continue
            try:
                conversation_index = int(conv_index_raw)
            except (TypeError, ValueError):
                continue
            try:
                positive_count = int(pos_raw) if pos_raw is not None else 0
            except (TypeError, ValueError):
                positive_count = 0
            try:
                total_messages = int(total_raw) if total_raw is not None else 0
            except (TypeError, ValueError):
                total_messages = 0
            key: ConversationKey = (
                participant,
                transcript_rel_path,
                conversation_index,
            )
            if key in mapping:
                continue
            conv_key = (conv_key_raw or "").strip() or None
            mapping[key] = ConversationCountsRow(
                participant=participant,
                transcript_rel_path=transcript_rel_path,
                conversation_index=conversation_index,
                conversation_key=conv_key,
                conversation_date=date_value,
                positive_count=positive_count,
                total_messages_in_run_for_conv=total_messages,
            )
    return mapping


def _collect_positive_instances_by_conversation(
    family_files: Sequence[Path],
    *,
    annotation_id: str,
    score_cutoff: Optional[int],
    conversations: dict[ConversationKey, ConversationCountsRow],
) -> dict[ConversationKey, list[PositiveInstance]]:
    """Return positive instances keyed by conversation, limited to ``conversations``."""

    positives: dict[ConversationKey, list[PositiveInstance]] = {}
    for record in iter_annotation_records(family_files, annotation_id=annotation_id):
        if not should_count_positive(record, score_cutoff=score_cutoff):
            continue
        fields = extract_conversation_key(record)
        if fields is None:
            continue
        participant, transcript_rel_path, conversation_index, _chat_key, _chat_date = (
            fields
        )
        key: ConversationKey = (participant, transcript_rel_path, conversation_index)
        if key not in conversations:
            continue
        try:
            msg_index_raw = record.get("message_index")
            msg_index = int(msg_index_raw)
        except (TypeError, ValueError):
            continue

        # Prefer the first non-empty quote from the model's ``matches`` field,
        # which is guaranteed to be an exact substring of the target message.
        quote_text: str = ""
        matches_raw = record.get("matches")
        if isinstance(matches_raw, list):
            for candidate in matches_raw:
                if isinstance(candidate, str) and candidate.strip():
                    quote_text = candidate.strip()
                    break

        if not quote_text:
            content_raw = record.get("content")
            quote_text = str(content_raw) if content_raw is not None else ""

        positives.setdefault(key, []).append(
            PositiveInstance(message_index=msg_index, content=quote_text)
        )
    return positives


def _build_quote_snippet(content: str, max_length: int = 200) -> str:
    """Return a short snippet suitable for the ``quote`` field.

    The returned text preserves the original character sequence so that it
    remains an exact substring of the underlying message content. Only a
    simple prefix slice is applied when truncation is required; no whitespace
    normalization or other transformations are performed.
    """

    text = content
    if len(text) <= max_length:
        return text
    return text[:max_length]


def _build_plan_rows(
    conversations: dict[ConversationKey, ConversationCountsRow],
    positives_by_conv: dict[ConversationKey, list[PositiveInstance]],
    *,
    label: str,
    prev_count: int,
    after_count: int,
    annotation_id: str,
    score_cutoff: Optional[int],
) -> list[dict[str, object]]:
    """Return plan CSV rows describing one subset per conversation."""

    rows: list[dict[str, object]] = []
    for key, conv_row in conversations.items():
        positives = positives_by_conv.get(key)
        if not positives:
            continue
        positives_sorted = sorted(positives, key=lambda item: item.message_index)
        anchor = positives_sorted[0]
        quote = _build_quote_snippet(anchor.content)
        plan_row: dict[str, object] = {
            PLAN_COLUMN_REL_PATH: conv_row.transcript_rel_path,
            PLAN_COLUMN_CONVERSATION_ID: conv_row.conversation_key or "",
            PLAN_COLUMN_QUOTE: quote,
            PLAN_COLUMN_LABEL: label,
            PLAN_COLUMN_PARTICIPANT: conv_row.participant,
            PLAN_COLUMN_PREV_COUNT: prev_count,
            PLAN_COLUMN_AFTER_COUNT: after_count,
            "comments": (
                f"Auto-generated from {annotation_id} conversation counts "
                f"(positive_count={conv_row.positive_count}, "
                f"score_cutoff={score_cutoff!r})."
            ),
            "annotation_id": annotation_id,
            "score_cutoff": score_cutoff,
            "conversation_index": conv_row.conversation_index,
            "conversation_date": conv_row.conversation_date,
            "positive_count": conv_row.positive_count,
            "total_messages_in_run_for_conv": conv_row.total_messages_in_run_for_conv,
        }
        rows.append(plan_row)
    return rows


def _write_plan_csv(rows: Iterable[dict[str, object]], output_csv: Path) -> None:
    """Write subset plan rows to ``output_csv``."""

    fieldnames = [
        PLAN_COLUMN_REL_PATH,
        PLAN_COLUMN_CONVERSATION_ID,
        PLAN_COLUMN_QUOTE,
        PLAN_COLUMN_LABEL,
        PLAN_COLUMN_PARTICIPANT,
        PLAN_COLUMN_PREV_COUNT,
        PLAN_COLUMN_AFTER_COUNT,
        "comments",
        "annotation_id",
        "score_cutoff",
        "conversation_index",
        "conversation_date",
        "positive_count",
        "total_messages_in_run_for_conv",
    ]

    write_dicts_to_csv(output_csv, fieldnames, rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point for the plan builder."""

    args = _parse_args(argv)

    resolved_counts = args.conversation_counts_csv.expanduser().resolve()
    if not resolved_counts.exists():
        print(f"Conversation counts CSV not found: {resolved_counts}")
        return 2

    # Infer score cutoff from the conversation-counts basename when not
    # explicitly provided. This keeps the CLI lightweight by reusing the
    # parameters encoded via ``dict_to_string`` in the counts script.
    inferred_cutoff: Optional[int] = None
    if args.score_cutoff is None:
        stem = resolved_counts.stem
        params = string_to_dict(stem)
        raw_cutoff = params.get("score_cutoff")
        if isinstance(raw_cutoff, int):
            inferred_cutoff = raw_cutoff

    resolved_reference = args.file.expanduser().resolve()
    if not resolved_reference.exists():
        print(f"Reference annotation JSONL not found: {resolved_reference}")
        return 2

    family_files = collect_family_files(resolved_reference, args.outputs_root)
    if not family_files:
        resolved_root = args.outputs_root.expanduser().resolve()
        print(
            f"No sibling files with basename {resolved_reference.name!r} "
            f"found under {resolved_root}",
        )
        return 2

    conversations = _load_conversation_counts(resolved_counts)
    if not conversations:
        print(f"No usable rows found in conversation counts CSV: {resolved_counts}")
        return 0

    positives_by_conv = _collect_positive_instances_by_conversation(
        family_files,
        annotation_id=str(args.annotation_id),
        score_cutoff=(
            args.score_cutoff if args.score_cutoff is not None else inferred_cutoff
        ),
        conversations=conversations,
    )
    plan_rows = _build_plan_rows(
        conversations,
        positives_by_conv,
        label=str(args.label),
        prev_count=int(args.prev_count),
        after_count=int(args.after_count),
        annotation_id=str(args.annotation_id),
        score_cutoff=(
            args.score_cutoff if args.score_cutoff is not None else inferred_cutoff
        ),
    )
    if not plan_rows:
        print("No conversations with positive messages matched the provided filters.")
        return 0

    if args.output_plan_csv is not None:
        output_csv = args.output_plan_csv
    else:
        output_csv = _build_default_plan_path(resolved_counts)

    _write_plan_csv(plan_rows, output_csv=output_csv)
    print(
        f"Wrote {len(plan_rows)} subset plan rows for "
        f"{len(positives_by_conv)} conversations to {output_csv}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
