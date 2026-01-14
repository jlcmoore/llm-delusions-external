"""Backfill preceding context into existing annotation JSONL outputs.

This helper script reads a manual annotation input JSONL file containing
``type: "item"`` records with ``preceding`` context and propagates that
context back into one or more ``classify_chats.py`` JSONL outputs from an
earlier run.

The script is intended for one-off JSONL "surgery" when older annotation
outputs were generated before ``preceding`` fields were added.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from annotation.io import (
    ParticipantMessageKey,
    build_participant_message_key,
    iter_jsonl_dicts_ignoring_errors,
)
from utils.io import parse_json_object_line

ManualKey = ParticipantMessageKey


def load_manual_preceding(
    manual_path: Path,
) -> Dict[ManualKey, List[Mapping[str, object]]]:
    """Return a mapping from message keys to preceding context from a manual JSONL.

    Parameters
    ----------
    manual_path:
        Path to a manual annotation input JSONL file containing ``type: "item"``
        records with ``preceding`` context.

    Returns
    -------
    Dict[ManualKey, List[Mapping[str, object]]]
        Mapping from ``(participant, source_path, chat_index, message_index)``
        keys to the corresponding ``preceding`` lists. When multiple items
        reference the same message key, the first non-empty ``preceding`` list
        is retained.
    """

    mapping: Dict[ManualKey, List[Mapping[str, object]]] = {}

    for obj in iter_jsonl_dicts_ignoring_errors(manual_path):
        if obj.get("type") != "item":
            continue

        key = build_participant_message_key(obj)
        if key is None:
            continue

        preceding_value = obj.get("preceding")
        if not isinstance(preceding_value, list) or not preceding_value:
            continue

        if key not in mapping:
            # Store the raw list; downstream consumers can normalize as needed.
            mapping[key] = list(preceding_value)

    return mapping


def iter_annotation_paths(
    participants: Iterable[str],
    run_basename: str,
    *,
    output_root: Path,
) -> List[Path]:
    """Return candidate annotation JSONL paths for a given run and participants.

    Parameters
    ----------
    participants:
        Iterable of participant identifiers, such as ``"hl_01"``.
    run_basename:
        Filename of the original annotation run JSONL (for example,
        ``"20251130-171532__input=...jsonl"``).
    output_root:
        Root directory containing annotation outputs, typically
        ``annotation_outputs/human_line``.

    Returns
    -------
    List[Path]
        Paths to existing JSONL files that will be considered for backfill.
    """

    paths: List[Path] = []
    for participant in participants:
        candidate = output_root / participant / run_basename
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
        else:
            logging.warning("Output JSONL not found for %s: %s", participant, candidate)
    return paths


def _build_manual_key(obj: Mapping[str, object]) -> Optional[ManualKey]:
    """Return the manual key for an annotation output record or None when invalid."""

    key = build_participant_message_key(obj)
    if key is None:
        return None
    return key


def backfill_preceding_into_file(
    jsonl_path: Path,
    mapping: Mapping[ManualKey, Sequence[Mapping[str, object]]],
) -> int:
    """Backfill preceding context into a single annotation JSONL file.

    Parameters
    ----------
    jsonl_path:
        Path to the annotation output JSONL to modify.
    mapping:
        Mapping from message keys to the desired ``preceding`` lists loaded
        from the manual inputs file.

    Returns
    -------
    int
        Number of records updated in ``jsonl_path``.
    """

    updated = 0
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")

    with (
        jsonl_path.open("r", encoding="utf-8", errors="ignore") as src,
        tmp_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            obj = parse_json_object_line(line)
            if obj is None:
                dst.write(line)
                continue

            key = _build_manual_key(obj)
            if key is not None and key in mapping:
                if not obj.get("preceding"):
                    obj["preceding"] = list(mapping[key])
                    updated += 1
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    shutil.move(str(tmp_path), str(jsonl_path))
    return updated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the backfill helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Backfill preceding context from a manual_annotation_inputs JSONL "
            "into one or more classify_chats annotation output JSONLs."
        )
    )
    parser.add_argument(
        "--manual-jsonl",
        type=Path,
        required=True,
        help="Path to the manual_annotation_inputs JSONL file.",
    )
    parser.add_argument(
        "--run-basename",
        required=True,
        help=(
            "Basename of the original annotation output JSONLs to modify "
            "(for example, '20251130-171532__input=...jsonl')."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("annotation_outputs") / "human_line",
        help=(
            "Root directory containing participant annotation outputs "
            "(default: annotation_outputs/human_line)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry point."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    manual_path = args.manual_jsonl.expanduser().resolve()
    if not manual_path.exists() or not manual_path.is_file():
        logging.error("Manual JSONL not found: %s", manual_path)
        return 2

    mapping = load_manual_preceding(manual_path)
    if not mapping:
        logging.error(
            "No usable preceding context found in manual JSONL: %s", manual_path
        )
        return 2

    participants = sorted({key[0] for key in mapping})
    output_root = args.output_root.expanduser().resolve()
    jsonl_paths = iter_annotation_paths(
        participants, args.run_basename, output_root=output_root
    )
    if not jsonl_paths:
        logging.error("No matching annotation JSONL files found under %s", output_root)
        return 2

    total_updated = 0
    for path in jsonl_paths:
        updated = backfill_preceding_into_file(path, mapping)
        logging.info("Updated %s records in %s", updated, path)
        total_updated += updated

    logging.info(
        "Backfill complete. Updated %s records across %s file(s).",
        total_updated,
        len(jsonl_paths),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
