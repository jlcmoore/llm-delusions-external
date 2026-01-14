"""Summarize subset quality and harmful annotations into a single CSV.

This script reads:

- ``subset_quality.jsonl`` produced by ``classify_subsets.py``.
- The subset JSON files under ``subsets/`` with attached per-message
  ``annotation_scores`` derived from the canonical annotations table.

It then writes a CSV aligned by subset row with:

- Subset-level quality scores from ``subset_quality.jsonl``.
- LLM notes and comments.
- Two additional flags derived from harmful annotations attached to subset
  messages:
  - ``has_harmful_annotations`` — any harmful label present in the subset.
  - ``harmful_annotations_early`` — at least one harmful label occurs within
    the first N user–assistant turns (configurable, default 50).

The output is intended to be pasted back into the Google Sheets plan to
improve and filter subsets, especially pivotal and harmful ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from annotation.annotation_prompts import ANNOTATIONS
from annotation.configs import LLM_SCORE_CUTOFF
from utils.io import iter_jsonl_dicts
from utils.schema import (
    MESSAGE_ROLE_KEY,
    RECORD_FIELD_COMMENTS,
    RECORD_FIELD_CONVERSATION_ID,
    RECORD_FIELD_CONVERSATION_TITLE,
    RECORD_FIELD_LABEL,
    RECORD_FIELD_LLM_NOTES,
    RECORD_FIELD_MESSAGES_COUNT,
    RECORD_FIELD_MODEL,
    RECORD_FIELD_PARTICIPANT,
    RECORD_FIELD_ROW,
    RECORD_FIELD_SCORES,
    RECORD_FIELD_SOURCE_REL_PATH,
    RECORD_FIELD_SUBSET_REL_PATH,
    RECORD_FIELD_TYPE,
    RECORD_TYPE_SUBSET_QUALITY,
    SCORE_FIELD_COHESION,
    SCORE_FIELD_PRIOR,
    SCORE_FIELD_UPLOADED,
    SUBSET_INFO_KEY,
    SUBSET_INFO_PARTICIPANT,
    SUBSET_MESSAGES_KEY,
    SUMMARY_COLUMN_COHESION,
    SUMMARY_COLUMN_COMMENTS,
    SUMMARY_COLUMN_EARLIEST_TURN,
    SUMMARY_COLUMN_HARMFUL_IDS,
    SUMMARY_COLUMN_HAS_HARMFUL,
    SUMMARY_COLUMN_LLM_NOTES,
    SUMMARY_COLUMN_PASSES_FILTERS,
    SUMMARY_COLUMN_PRIOR,
    SUMMARY_COLUMN_UPLOADED,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for subset harmfulness summarization."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarize subset quality scores and harmful annotation presence "
            "into a CSV aligned by subset row."
        )
    )
    parser.add_argument(
        "--subset-quality-json",
        type=Path,
        default=Path("subset_quality.jsonl"),
        help="Path to subset_quality.jsonl produced by classify_subsets.py.",
    )
    parser.add_argument(
        "--subsets-root",
        type=Path,
        default=Path("subsets"),
        help="Root directory containing subset JSON files (default: subsets).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("subset_quality_scores.csv"),
        help=(
            "Output CSV path containing per-row subset quality scores and "
            "harmfulness flags (default: subset_quality_scores.csv). The CSV "
            "is aligned to the plan CSV so rows can be pasted back into the "
            "original sheet."
        ),
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=Path("subsets.csv"),
        help=(
            "Original subsets plan CSV used to generate subsets. The output "
            "CSV will preserve this file's row order and base columns "
            "(default: subsets.csv)."
        ),
    )
    parser.add_argument(
        "--early-turn-threshold",
        type=int,
        default=50,
        help=(
            "Turn threshold for 'early' harmful annotations. A subset is "
            "marked harmful_annotations_early when the earliest harmful "
            "label occurs before this user–assistant turn index (default: 50)."
        ),
    )
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments and return the populated namespace."""

    parser = _build_parser()
    return parser.parse_args(argv)


def _load_subset_quality(
    path: Path,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, int]]:
    """Return per-row quality summaries and a mapping from subset path to row.

    Parameters
    ----------
    path:
        Path to ``subset_quality.jsonl``.

    Returns
    -------
    Tuple[Dict[int, Dict[str, object]], Dict[str, int]]
        A mapping from row number to a summary dictionary containing quality
        scores and metadata, and a mapping from subset_rel_path (relative to
        the subsets root) to row number.
    """

    by_row: Dict[int, Dict[str, object]] = {}
    subset_path_to_row: Dict[str, int] = {}

    for obj in iter_jsonl_dicts(path):
        if obj.get(RECORD_FIELD_TYPE) != RECORD_TYPE_SUBSET_QUALITY:
            continue
        row_value = obj.get(RECORD_FIELD_ROW)
        try:
            row = int(row_value)
        except (TypeError, ValueError):
            continue

        subset_rel_path = str(obj.get(RECORD_FIELD_SUBSET_REL_PATH) or "").strip()
        participant = obj.get(RECORD_FIELD_PARTICIPANT)
        label = obj.get(RECORD_FIELD_LABEL)
        source_rel_path = obj.get(RECORD_FIELD_SOURCE_REL_PATH)
        conversation_id = obj.get(RECORD_FIELD_CONVERSATION_ID)
        conversation_title = obj.get(RECORD_FIELD_CONVERSATION_TITLE)
        messages_count = obj.get(RECORD_FIELD_MESSAGES_COUNT)
        scores = obj.get(RECORD_FIELD_SCORES) or {}
        comments = obj.get(RECORD_FIELD_COMMENTS) or ""
        llm_notes_raw = obj.get(RECORD_FIELD_LLM_NOTES) or ""
        model_name = obj.get(RECORD_FIELD_MODEL)

        if llm_notes_raw and isinstance(llm_notes_raw, str):
            if isinstance(model_name, str) and model_name.strip():
                llm_notes = f"{llm_notes_raw}\n({model_name.strip()})"
            else:
                llm_notes = llm_notes_raw
        else:
            llm_notes = ""

        try:
            prior = int(scores.get(SCORE_FIELD_PRIOR))
            uploaded = int(scores.get(SCORE_FIELD_UPLOADED))
            cohesion = int(scores.get(SCORE_FIELD_COHESION))
        except (TypeError, ValueError):
            continue

        summary: Dict[str, object] = {
            RECORD_FIELD_ROW: row,
            RECORD_FIELD_PARTICIPANT: participant,
            RECORD_FIELD_LABEL: label,
            RECORD_FIELD_SUBSET_REL_PATH: subset_rel_path,
            RECORD_FIELD_SOURCE_REL_PATH: source_rel_path,
            RECORD_FIELD_CONVERSATION_ID: conversation_id,
            RECORD_FIELD_CONVERSATION_TITLE: conversation_title,
            RECORD_FIELD_MESSAGES_COUNT: messages_count,
            SCORE_FIELD_PRIOR: prior,
            SCORE_FIELD_UPLOADED: uploaded,
            SCORE_FIELD_COHESION: cohesion,
            SUMMARY_COLUMN_COMMENTS: comments,
            SUMMARY_COLUMN_LLM_NOTES: llm_notes,
        }
        by_row[row] = summary
        if subset_rel_path:
            subset_path_to_row[subset_rel_path] = row

    return by_row, subset_path_to_row


def _compute_turn_indices(
    messages: Sequence[Mapping[str, object]],
) -> List[Optional[int]]:
    """Return a list mapping message index to user–assistant turn index.

    The first user message receives turn index 0, the next user message 1,
    and so on. Assistant messages following a user message share that user's
    turn index. Messages before the first user message receive ``None``.
    """

    indices: List[Optional[int]] = []
    current_turn = -1
    for msg in messages:
        role_raw = msg.get(MESSAGE_ROLE_KEY)
        role = str(role_raw or "").strip().lower()
        if role == "user":
            current_turn += 1
        if current_turn < 0:
            indices.append(None)
        else:
            indices.append(current_turn)
    return indices


def _load_turn_index_cache(
    subsets_root: Path,
) -> Tuple[
    Dict[str, List[Optional[int]]],
    Dict[str, int],
]:
    """Return caches for turn indices and message counts per subset."""

    turn_cache: Dict[str, List[Optional[int]]] = {}
    length_cache: Dict[str, int] = {}
    root = subsets_root.expanduser().resolve()

    for file_path in root.rglob("*.json"):
        try:
            rel_path = str(file_path.relative_to(root))
        except ValueError:
            rel_path = str(file_path)
        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        messages = data.get(SUBSET_MESSAGES_KEY)
        if not isinstance(messages, list):
            continue
        turn_indices = _compute_turn_indices(messages)
        turn_cache[rel_path] = turn_indices
        length_cache[rel_path] = len(messages)
    return turn_cache, length_cache


def _load_harmful_annotation_ids() -> List[str]:
    """Return annotation ids flagged as harmful in the annotations registry."""

    harmful_ids: List[str] = []
    for spec in ANNOTATIONS:
        flag = str(spec.get("harmful_flag", "")).strip().lower()
        if flag in {"yes", "y", "true", "1", "harmful"}:
            ident = str(spec.get("id") or "").strip()
            if ident:
                harmful_ids.append(ident)
    return harmful_ids


def _summarize_harmful_annotations(
    subsets_root: Path,
    *,
    subset_path_to_row: Mapping[str, int],
    turn_cache: Mapping[str, List[Optional[int]]],
    early_turn_threshold: int,
) -> Tuple[Dict[int, Dict[str, object]], set[str]]:
    """Return per-row harmfulness summaries and participants with annotations.

    The participant set reflects all participants that have any attached
    ``annotation_scores`` in their subset messages, regardless of whether
    harmful labels are present above the LLM score cutoff.
    """

    harmful_ids = set(_load_harmful_annotation_ids())
    if not harmful_ids:
        logging.warning("No harmful annotations found in registry; skipping flags.")
        return {}, set()

    per_row: Dict[int, Dict[str, object]] = {}
    participants_with_outputs: set[str] = set()

    root = subsets_root.expanduser().resolve()

    for subset_rel_path, row in subset_path_to_row.items():
        file_path = root / subset_rel_path
        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        subset_info = data.get(SUBSET_INFO_KEY, {}) or {}
        participant_raw = subset_info.get(SUBSET_INFO_PARTICIPANT)
        participant = str(participant_raw or "").strip()

        messages = data.get(SUBSET_MESSAGES_KEY)
        if not isinstance(messages, list):
            continue

        turn_indices = turn_cache.get(subset_rel_path) or _compute_turn_indices(
            messages
        )

        for message_index, message in enumerate(messages):
            scores = message.get("annotation_scores")
            if not isinstance(scores, dict) or not scores:
                continue

            if participant:
                participants_with_outputs.add(participant)

            for key, raw_value in scores.items():
                if not str(key).startswith("score__"):
                    continue
                ann_id = str(key)[len("score__") :]
                if ann_id not in harmful_ids:
                    continue

                try:
                    llm_score = int(round(float(raw_value)))
                except (TypeError, ValueError, OverflowError):
                    continue
                if llm_score < LLM_SCORE_CUTOFF:
                    continue

                if message_index < 0 or message_index >= len(turn_indices):
                    continue
                turn_index = turn_indices[message_index]
                if turn_index is None:
                    continue

                summary = per_row.setdefault(
                    row,
                    {
                        "harmful_annotation_ids": [],
                        "earliest_harmful_turn": None,
                        "harmful_annotations_early": False,
                    },
                )

                ids_list = summary.get("harmful_annotation_ids") or []
                if ann_id not in ids_list:
                    ids_list.append(ann_id)
                summary["harmful_annotation_ids"] = ids_list
                existing = summary.get("earliest_harmful_turn")
                earliest: int
                if isinstance(existing, int):
                    earliest = min(existing, int(turn_index))
                else:
                    earliest = int(turn_index)
                summary["earliest_harmful_turn"] = earliest
                summary["harmful_annotations_early"] = bool(
                    earliest < early_turn_threshold
                )

    return per_row, participants_with_outputs


def _compute_passes_flag(
    label: str,
    prior: Optional[int],
    uploaded: Optional[int],
    cohesion: Optional[int],
    has_harmful_annotations: bool,
    harmful_annotations_early: bool,
    score_cutoff: int,
) -> Optional[bool]:
    """Return the passes_quality_filters flag for a single row.

    The flag is only computed when all five inputs are available:
    three quality scores plus harmful presence and early-ness.
    """

    if prior is None or uploaded is None or cohesion is None:
        return None

    normalized_label = (label or "").strip().lower()
    is_cohesive = cohesion >= score_cutoff
    low_prior = prior < score_cutoff
    low_uploaded = uploaded < score_cutoff
    harmful_combo = bool(has_harmful_annotations and harmful_annotations_early)

    # All labels must be cohesive and not over-reliant on uploads.
    if not is_cohesive or not low_uploaded:
        return False

    if normalized_label == "harmful":
        # Harmful subsets must also have an early harmful label.
        return harmful_combo

    if normalized_label == "pivotal":
        # Pivotal subsets must be cohesive, low prior, low uploads,
        # and have an early harmful label.
        if not low_prior:
            return False
        return harmful_combo

    # Normal (and any other) subsets must be cohesive, low prior, low uploads,
    # and must not simultaneously have a harmful label and have it early.
    if not low_prior:
        return False
    return not harmful_combo


def _write_summary_csv(
    plan_csv: Path,
    output_csv: Path,
    by_row: Mapping[int, Mapping[str, object]],
    harmful_by_row: Mapping[int, Mapping[str, object]],
    participants_without_outputs: set[str],
) -> None:
    """Write a consolidated CSV summary aligned to the plan CSV.

    The output preserves the original row order and base columns from the
    plan CSV, appending columns with subset quality scores and harmfulness
    flags. Rows without a corresponding subset_quality record receive empty
    values for the added columns.

    For participants with no annotation outputs under the annotation root,
    harmful-related columns are left blank rather than defaulting to False.
    """

    plan_path = plan_csv.expanduser().resolve()
    output_path = output_csv.expanduser().resolve()

    extra_columns = [
        SUMMARY_COLUMN_COMMENTS,
        SUMMARY_COLUMN_PRIOR,
        SUMMARY_COLUMN_UPLOADED,
        SUMMARY_COLUMN_COHESION,
        SUMMARY_COLUMN_HAS_HARMFUL,
        SUMMARY_COLUMN_HARMFUL_IDS,
        SUMMARY_COLUMN_EARLIEST_TURN,
        SUMMARY_COLUMN_PASSES_FILTERS,
        SUMMARY_COLUMN_LLM_NOTES,
    ]

    try:
        with (
            plan_path.open("r", encoding="utf-8", newline="") as src,
            output_path.open("w", encoding="utf-8", newline="") as dst,
        ):
            reader = csv.DictReader(src)
            base_fieldnames = reader.fieldnames or []
            fieldnames = list(base_fieldnames) + [
                name for name in extra_columns if name not in base_fieldnames
            ]

            writer = csv.DictWriter(
                dst,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()

            for idx, row in enumerate(reader, start=1):
                summary = by_row.get(idx)
                harmful = harmful_by_row.get(idx, {})
                participant = str(row.get("participant") or "").strip()

                if summary is not None:
                    row[SUMMARY_COLUMN_PRIOR] = summary.get(SCORE_FIELD_PRIOR, "")
                    row[SUMMARY_COLUMN_UPLOADED] = summary.get(SCORE_FIELD_UPLOADED, "")
                    row[SUMMARY_COLUMN_COHESION] = summary.get(SCORE_FIELD_COHESION, "")
                    row[SUMMARY_COLUMN_COMMENTS] = summary.get(
                        SUMMARY_COLUMN_COMMENTS, row.get(SUMMARY_COLUMN_COMMENTS, "")
                    )
                    row[SUMMARY_COLUMN_LLM_NOTES] = summary.get(
                        SUMMARY_COLUMN_LLM_NOTES, ""
                    )
                else:
                    for name in [
                        SUMMARY_COLUMN_PRIOR,
                        SUMMARY_COLUMN_UPLOADED,
                        SUMMARY_COLUMN_COHESION,
                        SUMMARY_COLUMN_LLM_NOTES,
                    ]:
                        row.setdefault(name, "")

                if participant and participant in participants_without_outputs:
                    row[SUMMARY_COLUMN_HAS_HARMFUL] = ""
                    row[SUMMARY_COLUMN_EARLIEST_TURN] = ""
                    row[SUMMARY_COLUMN_PASSES_FILTERS] = ""
                else:
                    raw_ids = harmful.get(SUMMARY_COLUMN_HARMFUL_IDS, [])
                    if isinstance(raw_ids, list):
                        harmful_ids = [str(value) for value in raw_ids if value]
                    else:
                        harmful_ids = []
                    has_harm = bool(harmful_ids)
                    earliest = harmful.get(SUMMARY_COLUMN_EARLIEST_TURN)
                    row[SUMMARY_COLUMN_HARMFUL_IDS] = (
                        ",".join(harmful_ids) if harmful_ids else ""
                    )
                    row[SUMMARY_COLUMN_EARLIEST_TURN] = (
                        int(earliest) if isinstance(earliest, int) else ""
                    )

                    if summary is not None:
                        label_value = (
                            summary.get(RECORD_FIELD_LABEL) or row.get("label") or ""
                        )
                        prior_val = summary.get(SCORE_FIELD_PRIOR)
                        uploaded_val = summary.get(SCORE_FIELD_UPLOADED)
                        cohesion_val = summary.get(SCORE_FIELD_COHESION)
                    else:
                        label_value = row.get("label") or ""
                        prior_val = uploaded_val = cohesion_val = None

                    passes_bool = _compute_passes_flag(
                        str(label_value),
                        int(prior_val) if isinstance(prior_val, int) else None,
                        int(uploaded_val) if isinstance(uploaded_val, int) else None,
                        int(cohesion_val) if isinstance(cohesion_val, int) else None,
                        has_harm,
                        bool(harmful.get("harmful_annotations_early", False)),
                        LLM_SCORE_CUTOFF,
                    )
                    row[SUMMARY_COLUMN_PASSES_FILTERS] = (
                        "" if passes_bool is None else passes_bool
                    )

                writer.writerow(row)
    except OSError as err:
        logging.error("Failed to write summary CSV %s: %s", output_path, err)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point for subset harmfulness summarization."""

    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    subset_quality_path = args.subset_quality_json.expanduser().resolve()
    if not subset_quality_path.exists() or not subset_quality_path.is_file():
        logging.error("subset_quality.jsonl not found: %s", subset_quality_path)
        return 2

    by_row, subset_path_to_row = _load_subset_quality(subset_quality_path)
    if not by_row:
        logging.error("No subset_quality records found in %s", subset_quality_path)
        return 2

    turn_cache, _length_cache = _load_turn_index_cache(args.subsets_root)
    harmful_by_row, participants_with_outputs = _summarize_harmful_annotations(
        args.subsets_root,
        subset_path_to_row=subset_path_to_row,
        turn_cache=turn_cache,
        early_turn_threshold=int(args.early_turn_threshold),
    )

    participants_in_quality = {
        str(summary.get("participant") or "").strip()
        for summary in by_row.values()
        if str(summary.get("participant") or "").strip()
    }
    participants_without_outputs = {
        p for p in participants_in_quality if p not in participants_with_outputs
    }
    if participants_without_outputs:
        logging.warning(
            "No annotation scores found in subsets for participants: %s",
            ", ".join(sorted(participants_without_outputs)),
        )

    _write_summary_csv(
        args.plan_csv,
        args.output_csv,
        by_row,
        harmful_by_row,
        participants_without_outputs,
    )
    logging.info("Wrote subset quality summary to %s", args.output_csv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
