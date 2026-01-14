"""Prepare repair inputs for reclassifying missing annotation scores.

This script inspects the canonical preprocessed per-message table
``annotations/all_annotations__preprocessed.parquet`` together with the
annotation metadata and identifies, for each message, annotation ids that
are in scope but have missing numeric scores. It then materialises one or
more LiteLLM batch manifests describing exactly these (message,
annotation_id) pairs so a follow-up ``classify_chats_batch.py enqueue``
and ``harvest`` run can reclassify them.

The script performs sanity checks to ensure that:

* Only in-scope annotations are considered missing.
* The number of repair rows matches the total number of missing
  in-scope (message, annotation) pairs.

Usage
-----

Run this once before launching a repair batch job:

.. code-block:: bash

    python scripts/annotation/build_missing_annotation_repair_inputs.py \
        --model gpt-5.1 \
        --preceding-context 3 \
        --batch-size 1000

This writes one or more manifest JSONL files in the ``analysis/data``
directory. The primary manifest is:

* ``analysis/data/all_annotations__missing_in_scope_annotations.jsonl``:
  a LiteLLM batch manifest containing up to ``--batch-size`` ``type:
  \"task\"`` records describing missing in-scope (message,
  annotation_id) pairs. When additional tasks are present, numbered
  ``__part-XXXX`` manifests are created alongside the primary file.
  All manifests can be passed directly to
  ``classify_chats_batch.py enqueue`` via the ``--manifest-dir`` flag.
  The ``job_name`` recorded in the manifest meta is ``all_annotations``
  so harvest will write outputs that share the original job stem.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from analysis_utils.annotation_metadata import (
    filter_analysis_metadata,
    load_annotation_metadata,
)
from analysis_utils.scope_coverage_utils import (
    build_in_scope_sets,
    compute_scope_coverage_counts,
    load_preprocessed_table,
    print_coverage_distribution,
)
from annotation.annotation_prompts import ANNOTATIONS, build_prompt
from annotation.batch_manifest import (
    ManifestConfig,
    create_manifest,
    encode_custom_id,
    update_manifest_status,
)
from llm_utils.client import DEFAULT_CHAT_MODEL
from utils.cli import (
    add_annotations_csv_argument,
    add_annotations_parquet_argument,
    add_output_path_argument,
)
from utils.io import get_default_transcripts_root, load_transcript_chats


def _compute_missing_pairs(
    frame: pd.DataFrame,
    *,
    scoped_ids_by_role: Mapping[str, Sequence[str]],
    non_test_ids: Sequence[str],
) -> Tuple[List[Dict[str, object]], Counter]:
    """Return missing in-scope (message, annotation_id) pairs.

    Parameters
    ----------
    frame:
        Preprocessed annotations DataFrame with ``score__<id>`` columns.
    scoped_ids_by_role:
        Mapping from role name to a list of annotation ids that apply to
        that role according to metadata scopes.
    non_test_ids:
        Sequence of non-test annotation ids present in the metadata.

    Returns
    -------
    rows:
        List of dictionaries, each containing ``participant``,
        ``source_path``, ``chat_index``, ``message_index``, ``role``, and
        ``annotation_id`` keys describing a missing in-scope annotation.
    coverage_counts:
        Counter keyed by ``(in_scope_count, non_nan_count)`` tuples
        summarising how many messages have the given coverage profile.
    """

    (
        in_scope_counts,
        non_nan_counts,
        coverage_counts,
        scoped_ids_by_role_filtered,
    ) = compute_scope_coverage_counts(frame, scoped_ids_by_role)

    roles = frame["role"].astype(str).str.lower().values
    missing_rows: List[Dict[str, object]] = []

    for idx, role_name in enumerate(roles):
        annotation_ids = scoped_ids_by_role_filtered.get(role_name, [])
        if not annotation_ids:
            continue
        in_scope_value = in_scope_counts[idx]
        non_nan_value = non_nan_counts[idx]
        if in_scope_value <= 0 or non_nan_value >= in_scope_value:
            continue

        row = frame.iloc[idx]
        scoped_score_columns = [
            f"score__{annotation_id}" for annotation_id in annotation_ids
        ]
        present_mask = row[scoped_score_columns].notna()
        for annotation_id, score_column in zip(annotation_ids, scoped_score_columns):
            if present_mask.get(score_column, False):
                continue
            missing_rows.append(
                {
                    "participant": row.get("participant", ""),
                    "source_path": row.get("source_path", ""),
                    "chat_index": int(row.get("chat_index", 0)),
                    "message_index": int(row.get("message_index", 0)),
                    "role": row.get("role", ""),
                    "annotation_id": annotation_id,
                }
            )

    expected_missing = sum(
        max(in_scope - non_nan, 0)
        for in_scope, non_nan in zip(in_scope_counts, non_nan_counts)
    )
    if expected_missing != len(missing_rows):
        raise RuntimeError(
            "Mismatch between expected and materialised missing pairs: "
            f"expected {expected_missing}, built {len(missing_rows)}.",
        )

    # Sanity-check that only non-test ids appear in the missing list.
    non_test_set = set(non_test_ids)
    unexpected_ids = {row["annotation_id"] for row in missing_rows} - non_test_set
    if unexpected_ids:
        raise RuntimeError(
            "Encountered unexpected annotation ids in missing-pairs list: "
            f"{sorted(unexpected_ids)}",
        )

    return missing_rows, coverage_counts


def _write_missing_pairs_jsonl(
    rows: Iterable[Mapping[str, object]],
    *,
    model: str,
    preceding_context: int,
    batch_size: int,
    base_manifest_path: Path,
    job_name: str = "all_annotations",
) -> int:
    """Write missing annotation pairs to sharded manifest JSONL repair files.

    Parameters
    ----------
    rows:
        Iterable of row dictionaries describing missing in-scope
        (message, annotation_id) pairs.
    model:
        Model name to record in the manifest meta.
    preceding_context:
        Number of preceding messages to include as context in prompts.
    batch_size:
        Maximum number of tasks to include per manifest JSONL file.
    base_manifest_path:
        Base path for the first manifest shard. Additional shards are
        created alongside this path with ``__part-XXXX`` suffixes.
    job_name:
        Logical job name to record in the manifest meta. Defaults to
        ``\"all_annotations\"`` so outputs share the original job stem.

    Returns
    -------
    int
        Number of task records written across all manifest files.
    """

    rows_list = list(rows)
    base_manifest_path = base_manifest_path.expanduser().resolve()
    base_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows_list:
        # Materialise an empty file so downstream tooling can still check
        # for its presence.
        base_manifest_path.write_text("", encoding="utf-8")
        return 0

    transcripts_root = get_default_transcripts_root()

    annotation_specs: Dict[str, Mapping[str, object]] = {
        str(item.get("id")): item for item in ANNOTATIONS
    }

    # Group rows by source_path and chat_index so that we only load each
    # transcript file once and can discard it after processing all of its
    # missing annotations.
    grouped_rows: Dict[str, Dict[int, List[Mapping[str, object]]]] = {}

    for row in rows_list:
        participant = str(row.get("participant") or "").strip()
        source_path = str(row.get("source_path") or "").strip()
        chat_index = int(row.get("chat_index", 0))
        message_index = int(row.get("message_index", 0))
        role = str(row.get("role") or "").strip()
        annotation_id = str(row.get("annotation_id") or "").strip()

        if not participant or not source_path or not annotation_id:
            continue
        annotation_spec = annotation_specs.get(annotation_id)
        if annotation_spec is None:
            continue
        grouped_for_source = grouped_rows.setdefault(source_path, {})
        grouped_for_chat = grouped_for_source.setdefault(chat_index, [])
        grouped_for_chat.append(
            {
                "participant": participant,
                "source_path": source_path,
                "chat_index": chat_index,
                "message_index": message_index,
                "role": role,
                "annotation_id": annotation_id,
            }
        )

    if not grouped_rows:
        base_manifest_path.write_text("", encoding="utf-8")
        return 0

    effective_batch_size = max(1, int(batch_size))
    written_tasks = 0
    shard_index = 0
    manifest_tasks: List[Dict[str, object]] = []

    total_rows = sum(
        len(rows_for_chat)
        for chats_by_index in grouped_rows.values()
        for rows_for_chat in chats_by_index.values()
    )

    def _flush_manifest_shard(
        shard_idx: int,
        shard_tasks: List[Dict[str, object]],
    ) -> None:
        """Write a single manifest shard and mark it pending."""

        if shard_idx == 0:
            manifest_path = base_manifest_path
        else:
            manifest_path = base_manifest_path.with_name(
                f"{base_manifest_path.stem}__part-{shard_idx:04d}.jsonl"
            )

        manifest_arguments: Dict[str, object] = {
            "model": model,
            "preceding_context": int(preceding_context),
            "repair_source": str(base_manifest_path),
        }
        manifest_config = ManifestConfig(
            job_name=job_name,
            batch_id="",
            input_file_id="",
            model=model,
            provider="openai",
            endpoint="/v1/chat/completions",
            arguments=manifest_arguments,
        )
        create_manifest(
            manifest_config,
            tasks=shard_tasks,
            manifest_path=manifest_path,
        )
        update_manifest_status(manifest_path, "pending")

    with tqdm(
        total=total_rows,
        desc="Building repair manifest tasks",
        unit="task",
    ) as progress:
        for source_path, chats_by_index in sorted(grouped_rows.items()):
            try:
                chats = load_transcript_chats(transcripts_root, source_path)
            except ValueError:
                progress.update(
                    sum(len(rows_for_chat) for rows_for_chat in chats_by_index.values())
                )
                continue

            for chat_index, rows_for_chat in sorted(chats_by_index.items()):
                if chat_index < 0 or chat_index >= len(chats):
                    raise ValueError(
                        f"chat_index {chat_index} out of range for {source_path}",
                    )
                messages = chats[chat_index].messages
                for row in rows_for_chat:
                    message_index = int(row.get("message_index", 0))
                    if message_index < 0 or message_index >= len(messages):
                        raise ValueError(
                            f"message_index {message_index} out of range for "
                            f"{source_path} chat_index {chat_index}",
                        )

                    participant = str(row.get("participant") or "").strip()
                    role = str(row.get("role") or "").strip()
                    annotation_id = str(row.get("annotation_id") or "").strip()
                    annotation_spec = annotation_specs.get(annotation_id)
                    if not participant or not annotation_id or annotation_spec is None:
                        progress.update(1)
                        continue

                    target = messages[message_index]
                    content = str(target.get("content") or "")
                    timestamp = str(target.get("timestamp") or "")

                    start_idx = max(0, message_index - int(preceding_context))
                    preceding_raw = messages[start_idx:message_index]
                    context_messages = [
                        {
                            "role": str(msg.get("role") or ""),
                            "content": str(msg.get("content") or ""),
                        }
                        for msg in preceding_raw
                        if str(msg.get("content") or "")
                    ]

                    prompt = build_prompt(
                        annotation_spec,
                        content,
                        role=role or None,
                        context_messages=context_messages,
                        include_cot_addendum=False,
                    )

                    custom_id = encode_custom_id(
                        participant,
                        source_path,
                        chat_index,
                        message_index,
                        annotation_id,
                    )

                    base_record: Dict[str, object] = {
                        "participant": participant,
                        "ppt_id": participant,
                        "source_path": source_path,
                        "chat_index": chat_index,
                        "chat_key": "",
                        "chat_date": "",
                        "message_index": message_index,
                        "role": role,
                        "timestamp": timestamp,
                    }
                    manifest_record: Dict[str, object] = {
                        "custom_id": custom_id,
                        **base_record,
                        "annotation_id": annotation_id,
                        "annotation_name": str(annotation_spec.get("name") or ""),
                        "prompt": prompt,
                        "content": content,
                        "preceding": context_messages,
                    }
                    manifest_tasks.append(manifest_record)
                    written_tasks += 1
                    progress.update(1)

                    if len(manifest_tasks) >= effective_batch_size:
                        _flush_manifest_shard(shard_index, manifest_tasks)
                        shard_index += 1
                        manifest_tasks = []

    if manifest_tasks:
        _flush_manifest_shard(shard_index, manifest_tasks)

    return int(written_tasks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the repair-input builder.

    Parameters
    ----------
    argv:
        Optional sequence of argument strings. When omitted, ``sys.argv``
        semantics are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace describing model, context depth, and
        batching behaviour.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build batch manifests of missing in-scope (message, annotation_id) "
            "pairs based on the canonical preprocessed annotations table."
        )
    )
    add_annotations_csv_argument(parser)
    add_annotations_parquet_argument(parser)
    add_output_path_argument(
        parser,
        default_path=Path("analysis")
        / "data"
        / "all_annotations__missing_in_scope_annotations.jsonl",
        help_text=(
            "Base path for the repair manifest JSONL file. Additional shards "
            "are created alongside this path with '__part-XXXX' suffixes."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CHAT_MODEL,
        help="Model name to record in the repair manifests.",
    )
    parser.add_argument(
        "--preceding-context",
        type=int,
        default=0,
        help=(
            "Number of preceding messages to include as context in repair prompts "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "Maximum number of (message, annotation_id) tasks to include per "
            "repair manifest JSONL file (default: 1000)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for building repair inputs for missing annotations.

    This function:

    * Loads non-test annotation metadata and scope information.
    * Loads the ``all_annotations__preprocessed.parquet`` table.
    * Computes, for each message, the number of in-scope annotations and
      the number of non-NaN scores.
    * Materialises one or more manifests with one row per missing
      in-scope (message, annotation_id) pair.
    * Prints a compact summary of coverage statistics and sanity checks.

    Returns
    -------
    int
        Zero on success, non-zero on failure.
    """

    args = parse_args(argv)

    annotations_csv = Path(args.annotations_csv)
    annotations_parquet = Path(args.annotations_parquet)
    base_manifest_path = Path(args.output)

    raw_metadata_by_id = load_annotation_metadata(annotations_csv)
    metadata_by_id = filter_analysis_metadata(raw_metadata_by_id)
    if not metadata_by_id:
        print("No non-test annotations discovered in metadata; nothing to repair.")
        return 0

    non_test_ids = sorted(metadata_by_id.keys())
    scoped_ids_by_role = build_in_scope_sets(metadata_by_id)

    frame = load_preprocessed_table(annotations_parquet)

    missing_rows, coverage_counts = _compute_missing_pairs(
        frame,
        scoped_ids_by_role=scoped_ids_by_role,
        non_test_ids=non_test_ids,
    )
    written_count = _write_missing_pairs_jsonl(
        missing_rows,
        model=args.model,
        preceding_context=args.preceding_context,
        batch_size=args.batch_size,
        base_manifest_path=base_manifest_path,
    )

    print(
        "Identified "
        f"{written_count} missing in-scope (message, annotation_id) pairs.",
    )
    print(
        "Wrote repair manifests under "
        f"{base_manifest_path.parent.expanduser().resolve()}",
    )

    total_messages = len(frame)
    print(f"Total messages in preprocessed table: {total_messages}")
    print_coverage_distribution(coverage_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
