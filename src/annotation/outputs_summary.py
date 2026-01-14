"""Shared helpers for summarizing annotation JSONL output families.

This module centralizes basic statistics used by multiple scripts when
inspecting classify_chats annotation outputs. Callers provide a list of
JSONL files that belong to a single job family and an outputs root; the
helpers then compute aggregate counts of rows, errors, quote mismatches,
and estimated tokens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from tqdm import tqdm

from annotation.io import iter_jsonl_meta, iter_jsonl_records


@dataclass
class OutputFamilyStats:
    """Aggregate statistics for a family of annotation output files.

    Parameters
    ----------
    total_rows:
        Total number of non-meta result rows across all files.
    total_errors:
        Number of rows with a non-empty ``error`` field.
    total_fatal_errors:
        Number of rows whose ``error`` does not represent a quote mismatch.
    total_quote_mismatch_errors:
        Number of rows whose ``error`` string reports a quote mismatch.
    total_estimated_tokens:
        Summed ``estimated_tokens`` value from meta headers when present.
    total_positive_rows:
        Number of rows whose ``score`` exceeds the configured cutoff.
    total_positive_rows_with_error:
        Number of positive rows that also have a non-empty ``error`` field.
    total_positive_rows_with_quote_mismatch_error:
        Number of positive rows whose error reports a quote mismatch.
    total_positive_rows_with_matches:
        Number of positive rows that contain at least one entry in ``matches``.
    fatal_error_messages:
        Counter of non-quote-mismatch error message texts.
    score_cutoff:
        Optional minimum score (0–10) for counting a row as positive. When
        omitted, scores greater than zero are treated as positive.
    """

    total_rows: int
    total_errors: int
    total_fatal_errors: int
    total_quote_mismatch_errors: int
    total_estimated_tokens: int
    total_positive_rows: int
    total_positive_rows_with_error: int
    total_positive_rows_with_quote_mismatch_error: int
    total_positive_rows_with_matches: int
    fatal_error_messages: Counter[str]
    score_cutoff: Optional[int]


def init_output_error_counters() -> Tuple[int, int, int, int, Counter[str]]:
    """Return zero-initialised counters for positive-row error statistics.

    Returns
    -------
    tuple
        A tuple containing ``(total_positive_rows, total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches, fatal_error_messages)``.
    """

    total_positive_rows = 0
    total_positive_rows_with_error = 0
    total_positive_rows_with_quote_mismatch_error = 0
    total_positive_rows_with_matches = 0
    fatal_error_messages: Counter[str] = Counter()
    return (
        total_positive_rows,
        total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches,
        fatal_error_messages,
    )


def _compute_output_family_stats_internal(
    family_files: Iterable[Path],
    *,
    outputs_root: Path,
    score_cutoff: Optional[int] = None,
    quote_mismatch_prefixes: Optional[Iterable[str]] = None,
) -> OutputFamilyStats:
    """Return aggregate statistics for a family of annotation outputs."""

    resolved_root = outputs_root.expanduser().resolve()
    quote_prefixes: Sequence[str]
    if quote_mismatch_prefixes is None:
        quote_prefixes = ("Quoted text not found in transcript",)
    else:
        quote_prefixes = tuple(str(prefix) for prefix in quote_mismatch_prefixes)

    total_rows = 0
    total_errors = 0
    total_fatal_errors = 0
    total_quote_mismatch_errors = 0
    total_estimated_tokens = 0
    (
        total_positive_rows,
        total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches,
        fatal_error_messages,
    ) = init_output_error_counters()

    # Load meta records once using the shared iterator and index by path.
    meta_by_path: dict[Path, dict] = {}
    for meta_path, meta in iter_jsonl_meta(resolved_root):
        meta_by_path[meta_path.resolve()] = meta

    for path in family_files:
        try:
            meta = meta_by_path.get(path.expanduser().resolve())
            if isinstance(meta, dict):
                arguments = meta.get("arguments") or {}
                estimated_tokens = arguments.get("estimated_tokens")
                if isinstance(estimated_tokens, (int, float)):
                    total_estimated_tokens += int(estimated_tokens)

            for obj in iter_jsonl_records(path):
                total_rows += 1

                raw_score = obj.get("score")
                is_positive = False
                if isinstance(raw_score, (int, float)):
                    numeric_score = int(raw_score)
                    if score_cutoff is not None:
                        is_positive = numeric_score >= score_cutoff
                    else:
                        is_positive = numeric_score > 0

                if is_positive:
                    total_positive_rows += 1
                    matches = obj.get("matches")
                    if isinstance(matches, list) and matches:
                        total_positive_rows_with_matches += 1

                error_message = obj.get("error")
                if not error_message:
                    continue

                total_errors += 1
                error_text = str(error_message)
                is_quote_mismatch = any(
                    error_text.startswith(prefix) for prefix in quote_prefixes
                )
                if is_quote_mismatch:
                    total_quote_mismatch_errors += 1
                    if is_positive:
                        total_positive_rows_with_error += 1
                        total_positive_rows_with_quote_mismatch_error += 1
                    continue

                total_fatal_errors += 1
                if is_positive:
                    total_positive_rows_with_error += 1
                fatal_error_messages[error_text] += 1
        except OSError:
            total_fatal_errors += 1
            fatal_error_messages[f"<os error while reading {path.name}>"] += 1

    return OutputFamilyStats(
        total_rows=total_rows,
        total_errors=total_errors,
        total_fatal_errors=total_fatal_errors,
        total_quote_mismatch_errors=total_quote_mismatch_errors,
        total_estimated_tokens=total_estimated_tokens,
        total_positive_rows=total_positive_rows,
        total_positive_rows_with_error=total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error=total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches=total_positive_rows_with_matches,
        fatal_error_messages=fatal_error_messages,
        score_cutoff=score_cutoff,
    )


def compute_output_family_stats(
    family_files: Sequence[Path],
    *,
    outputs_root: Path,
    score_cutoff: Optional[int] = None,
    quote_mismatch_prefixes: Optional[Iterable[str]] = None,
) -> OutputFamilyStats:
    """Return aggregate statistics for a family of annotation outputs.

    This variant performs a straightforward serial scan without progress
    reporting. It is suitable for non-interactive analysis helpers that need
    family-level statistics without user-facing feedback.
    """

    return _compute_output_family_stats_internal(
        family_files,
        outputs_root=outputs_root,
        score_cutoff=score_cutoff,
        quote_mismatch_prefixes=quote_mismatch_prefixes,
    )


def compute_output_family_stats_with_progress(
    family_files: Sequence[Path],
    *,
    outputs_root: Path,
    score_cutoff: Optional[int] = None,
    quote_mismatch_prefixes: Optional[Iterable[str]] = None,
) -> OutputFamilyStats:
    """Return family statistics while reporting progress over JSONL files.

    This helper mirrors the structure of other annotation scripts that scan
    a sorted job family using :class:`tqdm.tqdm` for user-visible progress.
    """

    iterator = tqdm(
        sorted(family_files),
        desc="Scanning annotation outputs",
        unit="file",
    )
    return _compute_output_family_stats_internal(
        iterator,
        outputs_root=outputs_root,
        score_cutoff=score_cutoff,
        quote_mismatch_prefixes=quote_mismatch_prefixes,
    )


__all__ = [
    "init_output_error_counters",
    "OutputFamilyStats",
    "compute_output_family_stats",
    "compute_output_family_stats_with_progress",
]
