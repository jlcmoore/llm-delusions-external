"""Re-check quote-mismatch errors against original transcripts.

This temporary diagnostic script scans a family of annotation output JSONL
files that share a basename (for example ``three_smallest.jsonl``),
identifies records whose ``error`` field reports quote-mismatch failures,
and then re-runs the canonical quote-matching logic against the original
message text loaded from transcripts.

The goal is to distinguish between:

- quotes that are genuinely absent from the target message content, and
- quotes that do appear in the transcript but were flagged as mismatched
  due to earlier pipeline issues (for example, missing or truncated
  ``content`` fields in batch-harvested outputs).

Usage example
-------------

    python scripts/annotation/recheck_quote_mismatches.py \\
        annotation_outputs/human_line/hl_04/fourth_batch.jsonl \\
        --transcripts-root transcripts_de_ided
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm

from annotation.classify_messages import filter_quotes_to_content, find_unmatched_quotes
from utils.cli import add_classify_chats_family_arguments, add_transcripts_root_argument
from utils.io import (
    collect_family_files,
    extract_message_location,
    get_simplified_messages,
    parse_json_object_line,
    warn_if_no_family_files,
)

FAILED_QUOTE_PREFIX = "Quoted text not found in transcript"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the recheck helper.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including the reference output file, outputs root,
        transcripts root, and optional limit on records to inspect.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Re-check quote-mismatch errors in annotation outputs against the "
            "original transcripts using the canonical quote-matching logic."
        )
    )
    add_classify_chats_family_arguments(parser, include_metadata=False)
    add_transcripts_root_argument(parser)
    parser.add_argument(
        "--max-error-records",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of error records with quote mismatches "
            "to inspect per file. When omitted, all such records in the family "
            "are processed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyze and report quote-mismatch fixes without modifying any "
            "annotation output files."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes to use when scanning annotation outputs. "
            "Set to 1 to disable parallelism (default: 1)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _extract_unmatched_quotes_from_error(error_message: str) -> List[str]:
    """Return a list of unmatched quotes parsed from an error string.

    The pipeline formats quote-mismatch fragments as:

    ``Quoted text not found in transcript: \"q1\", \"q2\", ...``

    This helper locates the prefix within the error message, then parses all
    double-quoted segments that follow.
    """

    start = error_message.find(FAILED_QUOTE_PREFIX)
    if start == -1:
        return []
    fragment = error_message[start + len(FAILED_QUOTE_PREFIX) :].lstrip()
    if fragment.startswith(":"):
        fragment = fragment[1:].lstrip()

    quotes: List[str] = []
    current: List[str] = []
    in_quotes = False
    for char in fragment:
        if char == '"':
            if in_quotes:
                quotes.append("".join(current))
                current = []
                in_quotes = False
            else:
                in_quotes = True
        else:
            if in_quotes:
                current.append(char)
    return quotes


RecordKey = Tuple[str, str, int, int, str]


def _scan_file_for_mismatches(
    jsonl_path: Path,
    transcripts_root: Path,
    max_error_records: Optional[int],
) -> Tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    dict[RecordKey, dict[str, object]],
    Counter[str],
]:
    """Return stats and planned updates for a single JSONL output file."""

    transcripts_root_resolved = transcripts_root.expanduser().resolve()
    total_records = 0
    total_error_records = 0
    total_error_records_processed = 0
    total_unmatched_quotes_reported = 0
    total_quotes_now_matched = 0
    total_quotes_still_unmatched = 0
    transcript_load_failures = 0

    message_cache: dict[Tuple[str, int], List[dict]] = {}
    recovered_examples: Counter[str] = Counter()
    planned_updates: dict[RecordKey, dict[str, object]] = {}

    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if FAILED_QUOTE_PREFIX not in stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("type") == "meta":
                    continue
                total_records += 1

                error_raw = record.get("error")
                if not isinstance(error_raw, str):
                    continue
                if FAILED_QUOTE_PREFIX not in error_raw:
                    continue

                total_error_records += 1
                if max_error_records is not None and (
                    total_error_records_processed >= max_error_records
                ):
                    continue

                unmatched_quotes = _extract_unmatched_quotes_from_error(error_raw)
                if not unmatched_quotes:
                    continue

                location = extract_message_location(record)
                if location is None:
                    continue
                source_path_raw, chat_index_int, message_index_int = location

                cache_key = (source_path_raw, chat_index_int)
                messages = message_cache.get(cache_key)
                if messages is None:
                    try:
                        messages = get_simplified_messages(
                            transcripts_root_resolved,
                            source_path_raw,
                            chat_index_int,
                        )
                    except (OSError, ValueError):
                        transcript_load_failures += 1
                        continue
                    message_cache[cache_key] = messages

                if message_index_int < 0 or message_index_int >= len(messages):
                    transcript_load_failures += 1
                    continue

                message_content = str(messages[message_index_int].get("content") or "")
                if not message_content:
                    transcript_load_failures += 1
                    continue

                total_error_records_processed += 1
                total_unmatched_quotes_reported += len(unmatched_quotes)

                newly_matched = filter_quotes_to_content(
                    unmatched_quotes,
                    message_content,
                )
                still_unmatched = find_unmatched_quotes(
                    unmatched_quotes,
                    message_content,
                )

                total_quotes_now_matched += len(newly_matched)
                total_quotes_still_unmatched += len(still_unmatched)

                if not newly_matched:
                    continue

                matches_field = record.get("matches")
                existing_matches: List[str] = []
                if isinstance(matches_field, list):
                    for item in matches_field:
                        if isinstance(item, str):
                            existing_matches.append(item)
                updated_matches = list(existing_matches)
                for quote in newly_matched:
                    if quote not in updated_matches:
                        updated_matches.append(quote)

                error_text = error_raw
                error_parts = [part for part in error_text.split(" | ") if part]
                base_parts: List[str] = []
                for part in error_parts:
                    trimmed = part.lstrip()
                    if trimmed.startswith(FAILED_QUOTE_PREFIX):
                        continue
                    base_parts.append(part)

                new_unmatched_fragment = None
                if still_unmatched:
                    new_unmatched_fragment = (
                        FAILED_QUOTE_PREFIX
                        + ": "
                        + ", ".join(f'"{quote}"' for quote in still_unmatched)
                    )
                    base_parts.append(new_unmatched_fragment)

                if base_parts:
                    updated_error: Optional[str] = " | ".join(base_parts)
                else:
                    updated_error = None

                record_key: RecordKey = (
                    str(record.get("participant") or ""),
                    source_path_raw,
                    chat_index_int,
                    message_index_int,
                    str(record.get("annotation_id") or ""),
                )
                planned_updates[record_key] = {
                    "matches": updated_matches,
                    "error": updated_error,
                }

                participant_raw = record.get("participant")
                participant = str(participant_raw or "")
                example_key = (
                    f"{participant}:{source_path_raw}:"
                    f"{chat_index_int}:{message_index_int}"
                )
                recovered_examples[example_key] += 1
    except OSError:
        transcript_load_failures += 1

    return (
        total_records,
        total_error_records,
        total_error_records_processed,
        total_unmatched_quotes_reported,
        total_quotes_now_matched,
        total_quotes_still_unmatched,
        transcript_load_failures,
        planned_updates,
        recovered_examples,
    )


def recheck_family(
    reference_file: Path,
    *,
    outputs_root: Path,
    transcripts_root: Path,
    max_error_records: Optional[int] = None,
    dry_run: bool = False,
    workers: int = 1,
) -> None:
    """Re-check quote-mismatch errors for a single job family.

    When ``dry_run`` is False, this function also applies safe, line-by-line
    updates to the affected JSONL files by promoting newly-matched quotes into
    the ``matches`` field and pruning the corresponding entries from the
    quote-mismatch error fragment. Files without any changes are left
    untouched.
    """

    family_files = collect_family_files(reference_file, outputs_root)
    if warn_if_no_family_files(family_files, reference_file, outputs_root):
        return

    transcripts_root_resolved = transcripts_root.expanduser().resolve()

    total_records = 0
    total_error_records = 0
    total_error_records_processed = 0
    total_unmatched_quotes_reported = 0
    total_quotes_now_matched = 0
    total_quotes_still_unmatched = 0
    transcript_load_failures = 0

    recovered_examples: Counter[str] = Counter()
    planned_updates_by_file: dict[Path, dict[RecordKey, dict[str, object]]] = {}

    # Phase 1: scan files (optionally in parallel) and plan updates.
    sorted_files = sorted(family_files)
    workers = max(1, int(workers or 1))
    if workers == 1:
        for jsonl_path in tqdm(sorted_files, desc="Scanning outputs", unit="file"):
            (
                recs,
                errs,
                errs_proc,
                unmatched_count,
                matched_count,
                still_unmatched_count,
                load_failures,
                planned,
                examples,
            ) = _scan_file_for_mismatches(
                jsonl_path,
                transcripts_root_resolved,
                max_error_records,
            )
            total_records += recs
            total_error_records += errs
            total_error_records_processed += errs_proc
            total_unmatched_quotes_reported += unmatched_count
            total_quotes_now_matched += matched_count
            total_quotes_still_unmatched += still_unmatched_count
            transcript_load_failures += load_failures
            if planned:
                file_path = jsonl_path.expanduser().resolve()
                planned_updates_by_file[file_path] = planned
            recovered_examples.update(examples)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(
                    _scan_file_for_mismatches,
                    path,
                    transcripts_root_resolved,
                    max_error_records,
                ): path
                for path in sorted_files
            }
            for future in tqdm(
                as_completed(future_to_path),
                total=len(future_to_path),
                desc="Scanning outputs",
                unit="file",
            ):
                path = future_to_path[future]
                (
                    recs,
                    errs,
                    errs_proc,
                    unmatched_count,
                    matched_count,
                    still_unmatched_count,
                    load_failures,
                    planned,
                    examples,
                ) = future.result()

                total_records += recs
                total_error_records += errs
                total_error_records_processed += errs_proc
                total_unmatched_quotes_reported += unmatched_count
                total_quotes_now_matched += matched_count
                total_quotes_still_unmatched += still_unmatched_count
                transcript_load_failures += load_failures
                if planned:
                    file_path = path.expanduser().resolve()
                    planned_updates_by_file[file_path] = planned
                recovered_examples.update(examples)

    print(
        "Re-check quote-mismatch results for "
        f"basename {reference_file.name!r} under outputs root {str(outputs_root)!r}"
    )
    print(f"  Total non-meta records scanned          : {total_records}")
    print(f"  Records with quote-mismatch errors      : {total_error_records}")
    print(
        "  Error records processed (limit-respecting): "
        f"{total_error_records_processed}"
    )
    print(
        "  Reported unmatched quotes in error fields: "
        f"{total_unmatched_quotes_reported}"
    )
    print(f"  Quotes now matched in transcripts       : {total_quotes_now_matched}")
    print(f"  Quotes still unmatched in transcripts   : {total_quotes_still_unmatched}")
    if total_unmatched_quotes_reported > 0:
        recovered_fraction = total_quotes_now_matched / float(
            total_unmatched_quotes_reported
        )
        print(
            "  Fraction of previously-unmatched quotes now matched: "
            f"{recovered_fraction:.3%}"
        )
    if transcript_load_failures:
        print(
            "  Records skipped due to transcript load or index issues: "
            f"{transcript_load_failures}"
        )

    if recovered_examples:
        print("\n  Example locations with newly matched quotes:")
        for key, count in recovered_examples.most_common(10):
            print(f"    {key}  ({count} recovered quote(s))")

    # Apply planned updates unless running in dry-run mode.
    if dry_run:
        if planned_updates_by_file:
            print(
                "\nDry run: planned updates exist but no files were modified. "
                "Rerun without --dry-run to apply changes."
            )
        return

    if not planned_updates_by_file:
        return

    print("\nApplying fixes to annotation output files...")

    updates_per_file: Counter[Path] = Counter()
    for jsonl_path in tqdm(
        sorted(family_files),
        desc="Rewriting updated outputs",
        unit="file",
    ):
        resolved_path = jsonl_path.expanduser().resolve()
        file_updates = planned_updates_by_file.get(resolved_path)
        if not file_updates:
            continue

        temp_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp_rechecked")
        file_changed = False

        try:
            with (
                resolved_path.open("r", encoding="utf-8", errors="ignore") as src,
                temp_path.open("w", encoding="utf-8") as dst,
            ):
                for line in src:
                    # Only lines that contain the quote-mismatch prefix can
                    # possibly correspond to planned updates.
                    if FAILED_QUOTE_PREFIX not in line:
                        dst.write(line)
                        continue

                    obj = parse_json_object_line(line)
                    if obj is None:
                        dst.write(line)
                        continue

                    try:
                        key_for_record: RecordKey = (
                            str(obj.get("participant") or ""),
                            str(obj.get("source_path") or ""),
                            int(obj.get("chat_index")),
                            int(obj.get("message_index")),
                            str(obj.get("annotation_id") or ""),
                        )
                    except (TypeError, ValueError):
                        dst.write(line)
                        continue

                    update = file_updates.get(key_for_record)
                    if not update:
                        dst.write(line)
                        continue

                    obj["matches"] = update.get("matches", obj.get("matches"))
                    if "error" in update:
                        obj["error"] = update["error"]

                    dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    file_changed = True
                    updates_per_file[resolved_path] += 1
        except OSError:
            # If we cannot rewrite a particular file, leave it untouched and
            # continue with others.
            continue

        if file_changed:
            try:
                temp_path.replace(resolved_path)
            except OSError:
                # If the atomic replace fails, fall back to leaving the
                # original file in place.
                continue
        else:
            try:
                temp_path.unlink()
            except OSError:
                pass

    if updates_per_file:
        print("\nUpdated files:")
        for path, count in updates_per_file.items():
            print(f"  {path}: {count} record(s) updated")


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Program entry point for the quote-mismatch recheck script."""

    args = parse_args(argv)
    reference_file = args.file.expanduser().resolve()
    outputs_root: Path = args.outputs_root
    resolved_outputs_root = outputs_root.expanduser().resolve()
    transcripts_root: Path = args.transcripts_root

    if not reference_file.exists():
        print(f"Reference file not found: {reference_file}")
        return 2
    if not resolved_outputs_root.exists() or not resolved_outputs_root.is_dir():
        print(
            "Outputs root not found or not a directory: " f"{resolved_outputs_root}",
        )
        return 2
    if not transcripts_root.exists() or not transcripts_root.is_dir():
        print(
            "Transcripts root not found or not a directory: " f"{transcripts_root}",
        )
        return 2

    recheck_family(
        reference_file,
        outputs_root=resolved_outputs_root,
        transcripts_root=transcripts_root,
        max_error_records=args.max_error_records,
        dry_run=bool(getattr(args, "dry_run", False)),
        workers=int(getattr(args, "workers", 1) or 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
