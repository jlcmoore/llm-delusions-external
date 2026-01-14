"""Aggregate positive annotation counts at the conversation level.

Given a single JSONL output file produced by ``classify_chats.py``, this script
discovers all sibling files with the same filename under the annotation output
root (for example, across different participants) and computes how many
messages in each conversation were positively flagged for a single annotation
identifier.

Each output row corresponds to a conversation locus keyed by
``(participant, transcript_rel_path, conversation_index)`` and reports the
number of positive messages alongside the total number of messages in the run
for that conversation and annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import quote

from annotation.annotation_prompts import should_count_positive
from annotation.conversation_counts import ConversationCountsRow
from annotation.io import extract_conversation_key, iter_annotation_records
from utils.cli import add_classify_chats_family_arguments, add_score_cutoff_argument
from utils.io import collect_family_files, warn_if_no_family_files, write_dicts_to_csv
from utils.param_strings import dict_to_string


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the conversation counter.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including the reference file, outputs root, target
        annotation id, score cutoff, minimum occurrences, and optional CSV
        output path.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate positive counts per conversation for a single annotation "
            "from classify_chats JSONL outputs."
        )
    )
    add_classify_chats_family_arguments(parser, include_metadata=False)
    parser.add_argument(
        "--annotation-id",
        "-a",
        required=True,
        help="Annotation identifier to aggregate at the conversation level.",
    )
    add_score_cutoff_argument(parser)
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help=(
            "Minimum number of positive messages required for a conversation "
            "to be included in the output (default: 1)."
        ),
    )
    parser.add_argument(
        "--max-positive-span",
        type=int,
        default=None,
        help=(
            "Optional maximum span in message indices between the earliest and "
            "latest positive instances within a conversation. When provided, "
            "only conversations whose positive messages fall within this many "
            "message indices are included."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional path for the output CSV. When omitted, a file is written "
            "under analysis/data/annotation_conversation_counts/ derived from the "
            "annotation id and score cutoff."
        ),
    )
    return parser.parse_args(argv)


def _build_default_output_path(
    *,
    annotation_id: str,
    score_cutoff: Optional[int],
    min_occurrences: int,
) -> Path:
    """Return a default CSV path based on aggregation parameters.

    The file is written under ``analysis/data/annotation_conversation_counts``
    so that downstream utilities can discover conversation-level aggregates
    without additional arguments.
    """

    base_dir = Path("analysis") / "data" / "annotation_conversation_counts"
    base_dir = base_dir.expanduser()
    params = {
        "annotation_id": annotation_id,
        "score_cutoff": score_cutoff if score_cutoff is not None else "any",
        "min_occurrences": min_occurrences,
    }
    filename = f"{dict_to_string(params)}.csv"
    return base_dir / filename


def compute_conversation_annotation_counts(
    family_files: Sequence[Path],
    *,
    annotation_id: str,
    score_cutoff: Optional[int],
    min_occurrences: int,
    max_positive_span: Optional[int] = None,
) -> list[ConversationCountsRow]:
    """Return aggregated annotation counts per conversation for the job family.

    Parameters
    ----------
    family_files:
        JSONL files that belong to the same classification job.
    annotation_id:
        Identifier of the annotation to aggregate.
    score_cutoff:
        Optional minimum score required for a message to count as positive.
    min_occurrences:
        Minimum number of positive messages required for a conversation to be
        included in the results.
    """

    stats_by_key: dict[tuple[str, str, int], ConversationCountsRow] = {}
    positive_indices: dict[tuple[str, str, int], list[int]] = {}

    for record in iter_annotation_records(family_files, annotation_id=annotation_id):
        fields = extract_conversation_key(record)
        if fields is None:
            continue
        participant, transcript_rel_path, conversation_index, chat_key, chat_date = (
            fields
        )
        key = (participant, transcript_rel_path, conversation_index)

        if key not in stats_by_key:
            stats_by_key[key] = ConversationCountsRow(
                participant=participant,
                transcript_rel_path=transcript_rel_path,
                conversation_index=conversation_index,
                conversation_key=chat_key,
                conversation_date=chat_date,
                positive_count=0,
                total_messages_in_run_for_conv=0,
            )
        stats = stats_by_key[key]

        stats.total_messages_in_run_for_conv += 1
        is_positive = should_count_positive(record, score_cutoff=score_cutoff)
        if is_positive:
            stats.positive_count += 1
            # Track message indices for span-based filtering when requested.
            try:
                msg_index_raw = record.get("message_index")
                msg_index = int(msg_index_raw)
            except (TypeError, ValueError):
                msg_index = None
            if msg_index is not None:
                bucket = positive_indices.setdefault(key, [])
                bucket.append(msg_index)

        if stats.conversation_key is None and chat_key:
            stats.conversation_key = chat_key
        if stats.conversation_date is None and chat_date is not None:
            stats.conversation_date = chat_date

    results: list[ConversationCountsRow] = []
    for key, stats in stats_by_key.items():
        if stats.positive_count <= 0:
            continue
        if stats.positive_count < min_occurrences:
            continue
        if max_positive_span is not None:
            indices = sorted(positive_indices.get(key, []))
            if not indices:
                # No reliable positions to evaluate span; skip this conversation.
                continue
            # Require that there exists a cluster of at least ``min_occurrences``
            # positive messages whose indices fall within ``max_positive_span``.
            window_ok = False
            required = max(min_occurrences, 1)
            n = len(indices)
            for start in range(n):
                end = start
                while end < n and indices[end] - indices[start] <= max_positive_span:
                    end += 1
                if end - start >= required:
                    window_ok = True
                    break
            if not window_ok:
                continue
        results.append(stats)

    results.sort(
        key=lambda item: (
            item.participant,
            item.transcript_rel_path,
            item.conversation_index,
        )
    )
    return results


def _write_conversation_counts_csv(
    rows: Sequence[ConversationCountsRow],
    *,
    output_csv: Path,
) -> None:
    """Write aggregated conversation counts to ``output_csv``."""

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "participant",
        "html_link",
        "transcript_rel_path",
        "conversation_index",
        "conversation_key",
        "conversation_date",
        "positive_count",
        "total_messages_in_run_for_conv",
    ]

    records: list[dict[str, object]] = []
    for stats in rows:
        rel_path = stats.transcript_rel_path
        if rel_path.endswith(".html.json"):
            html_rel = rel_path[: -len(".json")] + ".html"
        elif rel_path.endswith(".json"):
            html_rel = rel_path[: -len(".json")] + ".html"
        else:
            html_rel = rel_path + ".html"
        anchor_index = stats.conversation_index + 1
        raw_link = f"{html_rel}#chat-{anchor_index}"
        html_link = quote(raw_link, safe="/#")
        records.append(
            {
                "participant": stats.participant,
                "html_link": html_link,
                "transcript_rel_path": stats.transcript_rel_path,
                "conversation_index": stats.conversation_index,
                "conversation_key": stats.conversation_key or "",
                "conversation_date": stats.conversation_date,
                "positive_count": stats.positive_count,
                "total_messages_in_run_for_conv": stats.total_messages_in_run_for_conv,
            }
        )

    write_dicts_to_csv(output_csv, fieldnames, records)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Program entry point for the conversation-level counter."""

    args = parse_args(argv)

    resolved_reference = args.file.expanduser().resolve()
    if not resolved_reference.exists():
        print(f"Reference file not found: {resolved_reference}")
        return 2

    resolved_root = args.outputs_root.expanduser().resolve()

    family_files = collect_family_files(resolved_reference, args.outputs_root)
    if warn_if_no_family_files(family_files, resolved_reference, resolved_root):
        return 0

    min_occurrences = (
        int(args.min_occurrences) if args.min_occurrences is not None else 1
    )
    max_positive_span: Optional[int]
    if args.max_positive_span is None:
        max_positive_span = None
    else:
        max_positive_span = int(args.max_positive_span)
        if max_positive_span <= 0:
            max_positive_span = None

    stats = compute_conversation_annotation_counts(
        family_files,
        annotation_id=str(args.annotation_id),
        score_cutoff=args.score_cutoff,
        min_occurrences=min_occurrences,
        max_positive_span=max_positive_span,
    )
    if not stats:
        print(
            "No conversations with positive counts found for the requested "
            "annotation and cutoff.",
        )
        return 0

    if args.output_csv is not None:
        output_csv = args.output_csv
    else:
        output_csv = _build_default_output_path(
            annotation_id=str(args.annotation_id),
            score_cutoff=args.score_cutoff,
            min_occurrences=min_occurrences,
        )

    _write_conversation_counts_csv(stats, output_csv=output_csv)
    print(f"Wrote {len(stats)} conversation rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
