"""
Report preceding-context statistics for annotation JSONL outputs.

This helper scans JSONL files under ``annotation_outputs`` (or another root)
and summarizes how many preceding turns are attached to each classified
message. It is intended for debugging situations where the recorded
``preceding`` length in outputs does not match the expected
``preceding_context`` argument used at classification time.

Example
-------
To scan all ``all_annotations`` shards under the default outputs root::

    env-delusions/bin/python scripts/annotation/report_preceding_context.py \\
        --outputs-root annotation_outputs \\
        --name-contains all_annotations \\
        --threshold 3

This prints, for each matching file, the meta ``preceding_context`` value
along with per-participant counts of records whose ``preceding`` length
exceeds the given threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from utils.io import iter_jsonl_dicts


def _analyze_file(
    path: Path,
) -> Tuple[Optional[int], int, int, Mapping[str, Mapping[int, int]]]:
    """Return preceding statistics for a single JSONL output file.

    Parameters
    ----------
    path:
        Path to the JSONL file to analyze.
    threshold:
        Preceding length threshold used for downstream filtering.

    Returns
    -------
    tuple
        A tuple ``(meta_preceding, total_records, max_preceding,
        per_participant_counts)`` where:

        * ``meta_preceding`` is the integer ``preceding_context`` from the
          meta record when present, otherwise ``None``.
        * ``total_records`` is the number of non-meta records inspected.
        * ``max_preceding`` is the maximum ``len(preceding)`` observed.
        * ``per_participant_counts`` maps participant ids to a mapping from
          preceding length to record count.
    """

    meta_preceding: Optional[int] = None
    total_records = 0
    max_preceding = 0
    per_participant: Dict[str, Counter] = defaultdict(Counter)

    for obj in iter_jsonl_dicts(path):
        if obj.get("type") == "meta":
            if meta_preceding is None:
                preceding_raw = obj.get("preceding_context")
                if isinstance(preceding_raw, int):
                    meta_preceding = preceding_raw
            continue

        participant = str(obj.get("participant") or "").strip() or "<unknown>"
        preceding = obj.get("preceding")
        length = len(preceding) if isinstance(preceding, list) else 0

        total_records += 1
        max_preceding = max(max_preceding, length)
        per_participant[participant][length] += 1

    return meta_preceding, total_records, max_preceding, per_participant


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the reporting script.

    Parameters
    ----------
    argv:
        Optional custom argument list, primarily for testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments describing the outputs root and filters.
    """

    parser = argparse.ArgumentParser(
        description=("Summarize preceding-context lengths in annotation JSONL outputs.")
    )
    parser.add_argument(
        "--outputs-root",
        default="annotation_outputs",
        help=(
            "Root directory containing annotation JSONL files "
            "(default: annotation_outputs)."
        ),
    )
    parser.add_argument(
        "--name-contains",
        default=None,
        help=(
            "Optional substring filter applied to JSONL filenames. When "
            "provided, only files whose name contains this value are "
            "analyzed (for example, 'all_annotations')."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help=(
            "Highlight files and participants where len(preceding) exceeds "
            "this value (default: 3)."
        ),
    )
    parser.add_argument(
        "--examples-output",
        type=str,
        default=None,
        help=(
            "Optional path to a JSONL file where a sample of records with "
            "the maximum observed len(preceding) will be written. When "
            "omitted, no examples are saved."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """CLI entry point for the preceding-context reporting script.

    Parameters
    ----------
    argv:
        Optional custom argument list.

    Returns
    -------
    int
        Zero on success, non-zero on error.
    """

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = Path(args.outputs_root).expanduser().resolve()
    if not root.exists():
        logging.error("Outputs root %s does not exist.", root)
        return 2

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if args.name_contains:
        jsonl_files = [path for path in jsonl_files if args.name_contains in path.name]

    if not jsonl_files:
        logging.error(
            "No JSONL files found under %s with filter %r.",
            root,
            args.name_contains,
        )
        return 2

    logging.info(
        "Analyzing %d JSONL files under %s (filter=%r).",
        len(jsonl_files),
        root,
        args.name_contains,
    )

    # Aggregate statistics across all files rather than printing per-file
    # reports so that shards within a job family are summarized together.
    meta_histogram: Counter = Counter()
    global_len_counts: Counter = Counter()
    per_participant_over: Dict[str, int] = {}
    per_participant_max: Dict[str, int] = {}
    total_records = 0
    files_with_over_threshold = 0
    global_max_len = 0
    examples_output_path: Optional[Path] = (
        Path(args.examples_output).expanduser().resolve()
        if args.examples_output
        else None
    )
    # Track one example key per file for rows with the maximum observed
    # preceding length so we can write them out in a second pass.
    example_locations: Dict[Path, Dict[str, object]] = {}

    with ProcessPoolExecutor() as executor:
        future_to_path = {
            executor.submit(_analyze_file, path): path for path in jsonl_files
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                meta_preceding, count, max_len, per_ppt = future.result()
            except OSError as err:
                logging.error("Failed to analyze %s: %s", path, err)
                continue

            if count <= 0:
                continue

            if meta_preceding is not None:
                meta_histogram[int(meta_preceding)] += 1

            total_records += count
            global_max_len = max(global_max_len, max_len)

            if examples_output_path is not None and max_len > 0:
                # Record this file's maximum preceding length; the concrete
                # example is resolved in a second pass once the global
                # maximum is known.
                example_locations[path] = {
                    "max_len": max_len,
                }

            any_over_in_file = False
            for participant, counts in per_ppt.items():
                for length, c in counts.items():
                    global_len_counts[length] += c
                    if length > args.threshold:
                        any_over_in_file = True
                        per_participant_over[participant] = (
                            per_participant_over.get(participant, 0) + c
                        )
                        per_participant_max[participant] = max(
                            per_participant_max.get(participant, 0), length
                        )

            if any_over_in_file:
                files_with_over_threshold += 1

    if total_records == 0:
        logging.warning("No non-meta records were found in the selected files.")
        return 0

    print()
    print("Global preceding-context summary")
    print("--------------------------------")
    print(f"Total JSONL files scanned: {len(jsonl_files)}")
    print(f"Total non-meta records: {total_records}")
    print(f"Max len(preceding) observed: {global_max_len}")
    print(
        f"Files with any len(preceding) > {args.threshold}: "
        f"{files_with_over_threshold}"
    )

    if meta_histogram:
        print("\nMeta preceding_context values (per-file counts):")
        for value in sorted(meta_histogram.keys()):
            print(f"  {value} -> {meta_histogram[value]}")

    print("\nGlobal len(preceding) distribution (all records):")
    for length in sorted(global_len_counts.keys()):
        print(f"  {length} -> {global_len_counts[length]}")

    if per_participant_over:
        print(
            f"\nParticipants with len(preceding) > {args.threshold} "
            "(aggregated across files):"
        )
        sorted_participants = sorted(
            per_participant_over.items(), key=lambda item: item[1], reverse=True
        )
        for participant, over_count in sorted_participants[:20]:
            max_for_ppt = per_participant_max.get(participant, 0)
            print(
                f"  {participant}: {over_count} records above threshold, "
                f"max len={max_for_ppt}"
            )

    # Optionally materialize a small sample of concrete JSON records that
    # have the maximum observed preceding length for further inspection.
    if examples_output_path is not None and global_max_len > 0:
        written = 0
        with examples_output_path.open("w", encoding="utf-8") as handle:
            for path, info in example_locations.items():
                if int(info.get("max_len", 0)) != global_max_len:
                    continue
                for obj in iter_jsonl_dicts(path):
                    if obj.get("type") == "meta":
                        continue
                    preceding = obj.get("preceding")
                    if not isinstance(preceding, list):
                        continue
                    if len(preceding) != global_max_len:
                        continue
                    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                    # Keep a modest sample from each file to avoid bloating
                    # the examples output.
                    if written >= 100:
                        break
                if written >= 100:
                    break
        print(
            f"\nWrote up to 100 example records with "
            f"len(preceding) == {global_max_len} to {examples_output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
