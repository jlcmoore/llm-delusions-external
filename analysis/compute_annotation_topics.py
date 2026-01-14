"""Fit embedding-based global topics and per-annotation enrichment.

This script reads preprocessed match records produced by
``preprocess_annotation_family.py`` (via ``load_matches_records``), computes
sentence embeddings for each message using a BGE model, fits a global
background topic model with k-means over the embeddings, and then computes
per-annotation topic enrichment and heterogeneity statistics. All derived
artifacts are written under a dedicated analysis directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, TextIO

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from analysis_utils.annotation_tables import (
    LOCATION_WITH_CONTEXT_COLUMNS,
    build_content_mapping_for_locations,
)
from utils.cli import (
    Spinner,
    add_annotations_parquet_argument,
    add_artifacts_dir_argument,
    add_transcripts_parquet_argument,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the topic-modeling script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fit embedding-based global topics and per-annotation enrichment "
            "from a corpus defined by the preprocessed annotations table and "
            "transcript content."
        )
    )
    add_transcripts_parquet_argument(
        parser,
        help_text=(
            "Full transcripts Parquet table with message content. Used to "
            "look up content for messages referenced in the topic corpus "
            "without storing content in the matches artefacts."
        ),
    )
    add_annotations_parquet_argument(
        parser,
        help_text=(
            "Preprocessed per-message annotations Parquet table used as the "
            "canonical index for topic inputs. Matches are consulted only to "
            "assign messages to annotations."
        ),
    )
    add_artifacts_dir_argument(
        parser,
        default_dir=Path("analysis") / "data" / "annotation_topics_artifacts",
        help_text=(
            "Directory where topic-modeling artifacts (embeddings, topic "
            "summaries, and annotation enrichment tables) will be written."
        ),
    )
    parser.add_argument(
        "--num-topics",
        "-k",
        type=int,
        default=100,
        help=(
            "Number of global background topics to fit with k-means over "
            "message embeddings."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help=(
            "Optional maximum number of messages to use for testing. When "
            "provided, only the first N messages from the Parquet file are "
            "used for embeddings and topic modeling."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help=(
            "Sentence-transformers model name used to embed message content. "
            "Defaults to the BAAI bge-small-en-v1.5 model."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the topic-preparation script.

    Parameters
    ----------
    argv:
        Optional raw argument vector. When omitted, ``sys.argv`` semantics
        are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace populated with defaults.
    """

    parser = _build_parser()
    return parser.parse_args(argv)


def _load_annotation_rows(annotations_path: Path) -> List[Dict[str, object]]:
    """Return topic-input rows loaded from the annotations Parquet table.

    The annotations table is treated as the canonical per-message index and
    provides one row per message with location and lightweight metadata.
    Annotation-specific matches are resolved separately from the matches
    Parquet file and do not influence which messages enter the topic corpus.
    """

    ann_resolved = annotations_path.expanduser().resolve()
    if not ann_resolved.exists():
        raise FileNotFoundError(f"Annotations Parquet not found: {ann_resolved}")

    frame = pd.read_parquet(ann_resolved)
    if frame.empty:
        return []

    existing = [name for name in LOCATION_WITH_CONTEXT_COLUMNS if name in frame.columns]
    subset = frame[existing].copy()
    return subset.to_dict(orient="records")


def _attach_transcript_content(
    rows: List[Dict[str, object]],
    *,
    transcripts_path: Path,
) -> None:
    """Attach ``content`` strings to topic-input rows from transcripts data.

    This helper looks up message content in the ``transcripts.parquet`` table
    based on the standard location keys and mutates ``rows`` in place by
    adding a ``content`` field for each row. Messages without a matching
    transcript receive an empty string.
    """

    if not rows:
        return

    content_by_key = build_content_mapping_for_locations(transcripts_path, rows)

    for row in rows:
        key = (
            str(row.get("participant", "") or ""),
            str(row.get("source_path", "") or ""),
            int(row.get("chat_index", -1)),
            int(row.get("message_index", -1)),
        )
        row["content"] = content_by_key.get(key, "")


def _compute_or_load_embeddings(
    rows: Sequence[Mapping[str, object]],
    *,
    corpus_path: Path,
    artifacts_dir: Path,
    embedding_model_name: str,
) -> np.ndarray:
    """Return sentence embeddings for all messages, computing or loading cache.

    This function is responsible for caching the BGE embeddings so that they
    can be reused across repeated runs of the analysis without re-encoding all
    messages. The cache consists of:

    * ``embeddings.npz`` containing a compressed ``embeddings`` array and
      per-message location keys.
    * ``embeddings_meta.json`` describing the model, source path, counts, and
      cache layout.
    """

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = artifacts_dir / "embeddings.npz"
    meta_path = artifacts_dir / "embeddings_meta.json"

    resolved_corpus = corpus_path.expanduser().resolve()

    # Build stable per-message location keys used for incremental caching.
    row_keys = [
        (
            str(row.get("participant", "") or ""),
            str(row.get("source_path", "") or ""),
            int(row.get("chat_index", -1)),
            int(row.get("message_index", -1)),
        )
        for row in rows
    ]

    cached_embeddings: Optional[np.ndarray] = None
    cached_keys: Optional[List[tuple[str, str, int, int]]] = None
    meta: Dict[str, object] = {}

    if embeddings_path.exists() and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            meta = {}

        model_matches = meta.get("model_name") == embedding_model_name
        if model_matches:
            try:
                loaded = np.load(embeddings_path)
            except OSError:
                loaded = None

            if loaded is not None and "embeddings" in loaded.files:
                cached_embeddings = np.asarray(
                    loaded["embeddings"],
                    dtype=np.float32,
                )

                has_keys = all(
                    name in loaded.files
                    for name in (
                        "participant",
                        "source_path",
                        "chat_index",
                        "message_index",
                    )
                )
                if has_keys:
                    participant_keys = loaded["participant"]
                    source_path_keys = loaded["source_path"]
                    chat_index_keys = loaded["chat_index"]
                    message_index_keys = loaded["message_index"]

                    same_length = (
                        cached_embeddings.shape[0]
                        == len(participant_keys)
                        == len(source_path_keys)
                        == len(chat_index_keys)
                        == len(message_index_keys)
                    )
                    if same_length:
                        cached_keys = [
                            (
                                str(participant_keys[index]),
                                str(source_path_keys[index]),
                                int(chat_index_keys[index]),
                                int(message_index_keys[index]),
                            )
                            for index in range(len(participant_keys))
                        ]

    # If we have a cache with per-message keys, reuse embeddings where
    # possible and only compute new ones for unseen messages.
    embeddings: np.ndarray
    if cached_embeddings is not None and cached_keys is not None:
        key_to_index = {
            key: index for index, key in enumerate(cached_keys) if key is not None
        }
        embedding_dim = int(cached_embeddings.shape[1])
        embeddings = np.empty(
            (len(rows), embedding_dim),
            dtype=np.float32,
        )

        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for index, (key, row) in enumerate(zip(row_keys, rows)):
            cached_index = key_to_index.get(key)
            if cached_index is not None:
                embeddings[index] = cached_embeddings[cached_index]
            else:
                content = row.get("content")
                text_value = str(content) if content is not None else ""
                missing_indices.append(index)
                missing_texts.append(text_value)

        if missing_indices:
            print(
                f"Computing embeddings for {len(missing_indices)} new messages "
                f"using {embedding_model_name}...",
            )
            model = SentenceTransformer(embedding_model_name)
            new_embeddings = model.encode(
                missing_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            new_embeddings = np.asarray(new_embeddings, dtype=np.float32)

            for offset, row_index in enumerate(missing_indices):
                embeddings[row_index] = new_embeddings[offset]
        else:
            print("Reusing cached embeddings for all messages.")
    else:
        # No usable per-key cache; compute embeddings for all messages and
        # write a fresh cache in the new keyed format.
        texts: List[str] = []
        for row in rows:
            content = row.get("content")
            texts.append(str(content) if content is not None else "")

        print(
            f"Computing embeddings for {len(texts)} messages "
            f"using {embedding_model_name}...",
        )
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

    # Persist embeddings together with their location keys so that future
    # runs can reuse them even if the corpus changes slightly.
    participant_array = np.array([key[0] for key in row_keys])
    source_path_array = np.array([key[1] for key in row_keys])
    chat_index_array = np.array([key[2] for key in row_keys], dtype=np.int64)
    message_index_array = np.array([key[3] for key in row_keys], dtype=np.int64)

    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        participant=participant_array,
        source_path=source_path_array,
        chat_index=chat_index_array,
        message_index=message_index_array,
    )

    meta = {
        "model_name": embedding_model_name,
        "corpus_path": str(resolved_corpus),
        "num_messages": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
        "normalize_embeddings": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_version": 2,
        "key_fields": [
            "participant",
            "source_path",
            "chat_index",
            "message_index",
        ],
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return embeddings


def _compute_or_load_global_topics(
    rows: Sequence[Mapping[str, object]],
    embeddings: np.ndarray,
    *,
    artifacts_dir: Path,
    num_topics: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Return per-message topic assignments and a global topic summary.

    This function fits (or reloads) a k-means model over the sentence
    embeddings in order to define a shared background topic space for the
    entire corpus. It writes the following artifacts under ``artifacts_dir``:

    * ``message_topics.csv`` mapping message row indices to topic identifiers.
    * ``global_topics.json`` summarizing each topic with top terms and
      representative example messages.
    * ``global_topics_meta.json`` recording core parameters and counts.
    """

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    topics_meta_path = artifacts_dir / "global_topics_meta.json"
    topics_assignments_path = artifacts_dir / "message_topics.csv"
    topics_summary_path = artifacts_dir / "global_topics.json"

    num_messages = embeddings.shape[0]

    cached_assignments: Optional[np.ndarray] = None
    cached_summary: Dict[str, object] = {}

    if (
        topics_meta_path.exists()
        and topics_assignments_path.exists()
        and topics_summary_path.exists()
    ):
        try:
            with topics_meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            meta = {}

        topics_match = meta.get("num_topics") == num_topics
        messages_match = meta.get("num_messages") == num_messages
        tfidf_stop_words = meta.get("tfidf_stop_words")
        tfidf_min_df = meta.get("tfidf_min_df")
        tfidf_ngram_range = meta.get("tfidf_ngram_range")
        tfidf_matches = (
            tfidf_stop_words == "english"
            and tfidf_min_df == 5
            and tfidf_ngram_range == [1, 2]
        )

        if topics_match and messages_match and tfidf_matches:
            try:
                assignments = np.loadtxt(
                    topics_assignments_path,
                    delimiter=",",
                    skiprows=1,
                    dtype=int,
                    usecols=1,
                )
                if assignments.shape[0] == num_messages:
                    cached_assignments = assignments
            except OSError:
                cached_assignments = None

            try:
                with topics_summary_path.open("r", encoding="utf-8") as handle:
                    cached_summary = json.load(handle)
            except (OSError, json.JSONDecodeError):
                cached_summary = {}

    if cached_assignments is not None and cached_summary:
        return cached_assignments, cached_summary

    if num_topics <= 0:
        raise ValueError("num_topics must be a positive integer.")

    with Spinner(
        f"Fitting global k-means ({num_topics} topics) over "
        f"{num_messages} message embeddings..."
    ):
        kmeans = KMeans(
            n_clusters=num_topics,
            n_init=10,
            random_state=0,
        )
        topic_ids = kmeans.fit_predict(embeddings)

    # Prepare a TF-IDF representation for topic word summaries.
    texts: List[str] = []
    for row in rows:
        content = row.get("content")
        texts.append(str(content) if content is not None else "")

    with Spinner("Building TF-IDF matrix for topic labeling..."):
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=5,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    vocabulary = np.array(vectorizer.get_feature_names_out())

    topic_sizes = np.bincount(topic_ids, minlength=num_topics)

    num_examples_per_topic = 5
    top_terms_per_topic = 15

    topic_summaries: List[Dict[str, object]] = []

    for topic_index in tqdm(
        range(num_topics),
        desc="Global topics",
        unit="topic",
    ):
        indices_for_topic = np.where(topic_ids == topic_index)[0]
        size = int(topic_sizes[topic_index])

        if size == 0:
            topic_summaries.append(
                {
                    "topic_id": topic_index,
                    "size": 0,
                    "top_terms": [],
                    "top_term_scores": [],
                    "example_message_indices": [],
                }
            )
            continue

        # Compute mean TF-IDF weights within this topic and pick the top terms.
        topic_tfidf = tfidf_matrix[indices_for_topic].mean(axis=0)
        topic_tfidf_array = np.asarray(topic_tfidf).ravel()
        if topic_tfidf_array.size == 0:
            top_terms: List[str] = []
            top_term_scores: List[float] = []
        else:
            term_order = np.argsort(topic_tfidf_array)[::-1]
            top_indices = term_order[:top_terms_per_topic]
            filtered_indices = [
                int(i) for i in top_indices if topic_tfidf_array[i] > 0.0
            ]
            top_terms = [str(vocabulary[i]) for i in filtered_indices]
            top_term_scores = [float(topic_tfidf_array[i]) for i in filtered_indices]

        # Pick example messages nearest to the cluster center in embedding space.
        center = kmeans.cluster_centers_[topic_index]
        embeddings_for_topic = embeddings[indices_for_topic]
        scores = embeddings_for_topic @ center
        order = np.argsort(scores)[::-1]
        example_relative = order[:num_examples_per_topic]
        example_indices = [int(indices_for_topic[i]) for i in example_relative]

        topic_summaries.append(
            {
                "topic_id": topic_index,
                "size": size,
                "top_terms": top_terms,
                "top_term_scores": top_term_scores,
                "example_message_indices": example_indices,
            }
        )

    # Write assignments as a simple CSV with row index and topic id.
    with topics_assignments_path.open("w", encoding="utf-8") as handle:
        handle.write("row_index,topic_id\n")
        for index, topic_id in enumerate(topic_ids):
            handle.write(f"{index},{int(topic_id)}\n")

    global_summary: Dict[str, object] = {
        "num_topics": num_topics,
        "num_messages": int(num_messages),
        "topic_summaries": topic_summaries,
    }
    with topics_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(global_summary, handle, indent=2, sort_keys=True)

    meta: Dict[str, object] = {
        "num_topics": num_topics,
        "num_messages": int(num_messages),
        "tfidf_stop_words": "english",
        "tfidf_min_df": 5,
        "tfidf_ngram_range": [1, 2],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with topics_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return topic_ids, global_summary


def _compute_annotation_enrichment(
    rows: Sequence[Mapping[str, object]],
    topic_ids: np.ndarray,
    *,
    annotation_to_indices: Mapping[str, Sequence[int]],
    num_topics: int,
    artifacts_dir: Path,
) -> None:
    """Compute per-annotation topic enrichment and heterogeneity statistics.

    This function writes two CSV files under ``artifacts_dir``:

    * ``annotation_summary.csv`` with one row per annotation summarizing
      entropy, dominant topics, and message counts.
    * ``annotation_topic_enrichment.csv`` with one row per annotation-topic
      pair capturing relative enrichment compared with the global background.
    """

    if topic_ids.shape[0] != len(rows):
        raise ValueError("Topic assignment array length does not match rows.")

    total_messages = len(rows)
    if total_messages == 0:
        return

    global_counts = np.bincount(topic_ids, minlength=num_topics)

    summary_path = artifacts_dir / "annotation_summary.csv"
    enrichment_path = artifacts_dir / "annotation_topic_enrichment.csv"

    eps = 1e-9

    with (
        summary_path.open("w", encoding="utf-8") as summary_file,
        enrichment_path.open(
            "w",
            encoding="utf-8",
        ) as enrichment_file,
    ):
        summary_file.write(
            "annotation_id,num_messages,entropy,normalized_entropy,"
            "dominant_topic_id,dominant_topic_proportion\n"
        )
        enrichment_file.write(
            "annotation_id,topic_id,annotation_count,annotation_proportion,"
            "global_count,global_proportion,log_enrichment\n"
        )

        for annotation_id, indices in tqdm(
            sorted(annotation_to_indices.items()),
            desc="Annotations",
            unit="ann",
        ):
            _write_annotation_stats(
                annotation_id,
                indices,
                topic_ids,
                global_counts,
                total_messages,
                num_topics,
                summary_file=summary_file,
                enrichment_file=enrichment_file,
                eps=eps,
            )


def _write_annotation_stats(
    annotation_id: str,
    indices: Sequence[int],
    topic_ids: np.ndarray,
    global_counts: np.ndarray,
    total_messages: int,
    num_topics: int,
    *,
    summary_file: TextIO,
    enrichment_file: TextIO,
    eps: float,
) -> None:
    """Write entropy and enrichment statistics for a single annotation."""

    annotation_size = len(indices)
    if annotation_size == 0:
        return

    ann_counts = np.bincount(topic_ids[indices], minlength=num_topics)

    proportions = ann_counts.astype(np.float64) / float(annotation_size)
    mask = proportions > 0.0
    entropy = float(-np.sum(proportions[mask] * np.log(proportions[mask])))
    max_entropy = math.log(num_topics) if num_topics > 1 else 0.0
    if max_entropy > 0.0:
        normalized = float(entropy / max_entropy)
    else:
        normalized = 0.0

    dominant_topic_id = int(np.argmax(proportions))
    dominant_prop = float(proportions[dominant_topic_id])

    summary_file.write(
        f"{annotation_id},{annotation_size},{entropy},"
        f"{normalized},{dominant_topic_id},{dominant_prop}\n"
    )

    for topic_index in range(num_topics):
        ann_count = int(ann_counts[topic_index])
        if ann_count == 0:
            continue

        ann_prop = float(proportions[topic_index])
        global_count = int(global_counts[topic_index])
        global_prop = float(global_counts[topic_index] / float(total_messages))

        log_enrichment = math.log((ann_prop + eps) / (global_prop + eps))

        enrichment_file.write(
            f"{annotation_id},{topic_index},{ann_count},{ann_prop},"
            f"{global_count},{global_prop},{log_enrichment}\n"
        )


def _compute_participant_topics(
    rows: Sequence[Mapping[str, object]],
    topic_ids: np.ndarray,
    *,
    num_topics: int,
    artifacts_dir: Path,
) -> None:
    """Compute per-participant topic distributions and enrichment.

    This function writes a CSV file under ``artifacts_dir`` with one row per
    (participant, topic) pair containing counts, per-participant topic
    proportions, global topic proportions, and log-enrichment values.
    """

    if topic_ids.shape[0] != len(rows):
        raise ValueError("Topic assignment array length does not match rows.")

    total_messages = len(rows)
    if total_messages == 0:
        return

    global_counts = np.bincount(topic_ids, minlength=num_topics)

    participant_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        participant = str(row.get("participant", "") or "")
        if participant:
            participant_to_indices[participant].append(index)

    output_path = artifacts_dir / "participant_topics.csv"

    eps = 1e-9

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "participant,topic_id,participant_topic_count,"
            "participant_total_messages,participant_topic_proportion,"
            "global_topic_count,global_topic_proportion,log_enrichment\n"
        )

        for participant, indices in sorted(participant_to_indices.items()):
            total_for_participant = len(indices)
            if total_for_participant == 0:
                continue

            participant_topic_ids = topic_ids[indices]
            counts = np.bincount(participant_topic_ids, minlength=num_topics)

            for topic_index in range(num_topics):
                count = int(counts[topic_index])
                if count == 0:
                    continue

                prop = float(count) / float(total_for_participant)
                global_count = int(global_counts[topic_index])
                global_prop = float(global_counts[topic_index] / float(total_messages))

                log_enrichment = math.log((prop + eps) / (global_prop + eps))

                handle.write(
                    f"{participant},{topic_index},{count},"
                    f"{total_for_participant},{prop},"
                    f"{global_count},{global_prop},{log_enrichment}\n"
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point for embedding-based topic modeling."""

    args = parse_args(argv)

    # Load the canonical per-message corpus from the annotations table.
    topic_inputs = _load_annotation_rows(args.annotations_parquet)

    if args.max_messages is not None and args.max_messages > 0:
        topic_inputs = topic_inputs[: args.max_messages]

    # Attach message content from transcripts_data so that embeddings and
    # topic summaries rely on a single canonical transcripts store rather
    # than duplicating content in the matches artefacts.
    _attach_transcript_content(
        topic_inputs,
        transcripts_path=args.transcripts_parquet,
    )

    print(f"Loaded {len(topic_inputs)} topic-input messages from annotations.")

    # Compute and cache embeddings for all messages, then fit global topics and
    # compute per-annotation enrichment statistics. All artifacts are written
    # under the configured artifacts directory.
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    embeddings = _compute_or_load_embeddings(
        topic_inputs,
        corpus_path=args.annotations_parquet,
        artifacts_dir=artifacts_dir,
        embedding_model_name=args.embedding_model,
    )

    topic_ids, _ = _compute_or_load_global_topics(
        topic_inputs,
        embeddings,
        artifacts_dir=artifacts_dir,
        num_topics=args.num_topics,
    )
    # Per-annotation enrichment currently does not assign messages to
    # annotations; matches are reserved for downstream interpretation of
    # clusters (see PAPER_TODO). The mapping is left empty so that only
    # global topic structure is summarised.
    annotation_to_indices: Dict[str, List[int]] = {}

    _compute_annotation_enrichment(
        topic_inputs,
        topic_ids,
        annotation_to_indices=annotation_to_indices,
        num_topics=args.num_topics,
        artifacts_dir=artifacts_dir,
    )

    _compute_participant_topics(
        topic_inputs,
        topic_ids,
        num_topics=args.num_topics,
        artifacts_dir=artifacts_dir,
    )

    print(
        "Global topic modeling artifacts written under " f"{artifacts_dir}",
    )
    print(
        "Per-annotation topic enrichment statistics are available in "
        f"{(artifacts_dir / 'annotation_summary.csv').expanduser().resolve()} "
        "and "
        f"{(artifacts_dir / 'annotation_topic_enrichment.csv').expanduser().resolve()}",
    )
    print(
        "Per-participant topic statistics are available in "
        f"{(artifacts_dir / 'participant_topics.csv').expanduser().resolve()}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
