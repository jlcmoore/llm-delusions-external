"""Compute windowed incidence counts for a single annotation.

This script inspects a preprocessed per-message annotation table and, for a
single annotation id, computes how often that annotation appears within
contiguous windows of size K inside each conversation. Occurrences are
grouped into clusters per conversation such that consecutive positive
message indices separated by at most K messages belong to the same window.
Larger gaps start a new window within the same conversation.

For all windows that contain at least one positive message, the script
reports:

* The total number of windows.
* The mean number of positive messages per window.
* The standard deviation and naive standard error over windows.
* A participant-clustered standard error that treats participants as
  independent clusters.

Conversations are defined by ``chat_index`` and never wrap across
conversation boundaries. Role scopes from the annotation metadata are
respected, and an optional role filter allows restricting the analysis to
user or assistant messages only.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from analysis_utils.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
    normalize_role_filter,
)
from analysis_utils.formatting import round3
from annotation.io import ParticipantMessageKey
from utils.cli import (
    add_annotation_metadata_arguments,
    add_output_path_argument,
    add_preprocessed_input_csv_argument,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the window-incidence script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute windowed incidence counts for a single annotation using "
            "a preprocessed per-message annotation table."
        )
    )
    add_preprocessed_input_csv_argument(parser)
    add_annotation_metadata_arguments(parser)
    parser.add_argument(
        "--annotation-id",
        required=True,
        help=(
            "Annotation id to analyse. Only messages where this annotation "
            "is positive and in scope contribute to windowed counts."
        ),
    )
    parser.add_argument(
        "--role-scope",
        type=str,
        default="auto",
        help=(
            "Optional role restriction applied on top of annotation scopes. "
            "When set to 'user' or 'assistant', only messages with that role "
            "contribute to windowed counts. When omitted or set to "
            "'auto'/'both', both roles are included wherever the annotation "
            "is in scope."
        ),
    )
    parser.add_argument(
        "--window-k",
        type=int,
        default=50,
        help=(
            "Maximum allowed gap in message indices between consecutive "
            "positive occurrences for them to belong to the same window "
            "(default: 50). Larger gaps start a new window within the same "
            "conversation."
        ),
    )
    add_output_path_argument(
        parser,
        default_path=Path("analysis") / "figures" / "next_k_counts_histogram.pdf",
        help_text=(
            "Output PDF path for the histogram of additional positive-message "
            "counts within the next K turns."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the window-incidence script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted, ``sys.argv``
        semantics are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments populated with defaults.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.window_k <= 0:
        parser.error("--window-k must be a positive integer")

    try:
        args.role_scope = normalize_role_filter(getattr(args, "role_scope", None))
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _collect_window_counts_for_annotation(
    annotation_id: str,
    *,
    message_info: Mapping[ParticipantMessageKey, Tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[Tuple[str, ParticipantMessageKey], bool],
    metadata_by_id: Mapping[str, AnnotationMetadata],
    window_k: int,
    role_scope: Optional[str],
) -> Tuple[list[int], list[str]]:
    """Return per-window positive counts and participant ids for an annotation.

    This helper groups positive occurrences of ``annotation_id`` within each
    conversation into contiguous windows where the difference between adjacent
    positive message indices is at most ``window_k``. Each such window is
    summarised by the number of positive messages it contains.

    Parameters
    ----------
    annotation_id:
        Identifier of the annotation to analyse.
    message_info:
        Mapping from participant message key to (role, conversation key).
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to positivity flags.
    metadata_by_id:
        Annotation metadata keyed by annotation id.
    window_k:
        Maximum allowed gap in message indices between consecutive positive
        occurrences for them to belong to the same window.
    role_scope:
        Optional role restriction (``"user"`` or ``"assistant"``) applied on
        top of the annotation scope. When ``None``, both roles are included
        wherever the annotation is in scope.

    Returns
    -------
    window_counts:
        List of positive-message counts per window.
    window_participants:
        Corresponding participant ids per window, used for clustering.
    """

    meta = metadata_by_id.get(annotation_id)
    if meta is None:
        return [], []
    scope = meta.scope

    occurrences_by_conversation: Dict[ConversationKey, list[int]] = defaultdict(list)

    for (aid, message_key), is_positive in annotation_message_positive.items():
        if aid != annotation_id or not is_positive:
            continue
        info = message_info.get(message_key)
        if info is None:
            continue
        role, conv_key = info
        if not is_role_in_scope(role, scope):
            continue
        if role_scope is not None and role != role_scope:
            continue
        try:
            message_index = int(message_key[3])
        except (TypeError, ValueError):
            continue
        occurrences_by_conversation[conv_key].append(message_index)

    window_counts: list[int] = []
    window_participants: list[str] = []

    for conv_key, indices in occurrences_by_conversation.items():
        if not indices:
            continue
        sorted_indices = sorted(indices)
        current_count = 1
        previous_index = sorted_indices[0]

        for index in sorted_indices[1:]:
            if index - previous_index <= window_k:
                current_count += 1
            else:
                window_counts.append(current_count)
                window_participants.append(conv_key.participant)
                current_count = 1
            previous_index = index

        window_counts.append(current_count)
        window_participants.append(conv_key.participant)

    return window_counts, window_participants


def _collect_next_k_counts_for_annotation(
    annotation_id: str,
    *,
    message_info: Mapping[ParticipantMessageKey, Tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[Tuple[str, ParticipantMessageKey], bool],
    metadata_by_id: Mapping[str, AnnotationMetadata],
    window_k: int,
    role_scope: Optional[str],
) -> list[int]:
    """Return per-occurrence counts within a fixed K-message window.

    For each positive occurrence of ``annotation_id`` on a message index ``i``,
    this helper counts the number of later messages in the same conversation
    whose indices lie in ``(i, i + window_k]`` and are also positive for the
    annotation. The result is a list of counts (one per occurrence) and the
    corresponding participant ids used for clustering.
    """

    meta = metadata_by_id.get(annotation_id)
    if meta is None:
        return [], []
    scope = meta.scope

    positive_indices_by_conv: Dict[ConversationKey, list[int]] = defaultdict(list)

    for (aid, message_key), is_positive in annotation_message_positive.items():
        if aid != annotation_id or not is_positive:
            continue
        info = message_info.get(message_key)
        if info is None:
            continue
        role, conv_key = info
        if not is_role_in_scope(role, scope):
            continue
        if role_scope is not None and role != role_scope:
            continue
        try:
            message_index = int(message_key[3])
        except (TypeError, ValueError):
            continue
        positive_indices_by_conv[conv_key].append(message_index)

    counts: list[int] = []

    for _conv_key, indices in positive_indices_by_conv.items():
        if not indices:
            continue
        sorted_indices = sorted(indices)
        n_indices = len(sorted_indices)
        j = 0
        for i in range(n_indices):
            j = max(j, i + 1)
            while j < n_indices and sorted_indices[j] - sorted_indices[i] <= window_k:
                j += 1
            count_in_window = j - (i + 1)
            counts.append(count_in_window)

    return counts


def _plot_window_counts_boxplot(
    window_counts: Sequence[int],
    annotation_id: str,
    role_scope: Optional[str],
    window_k: int,
    output_path: Path,
) -> None:
    """Write a box-and-whisker + scatter + median plot of window counts.

    Parameters
    ----------
    window_counts:
        Sequence of positive-message counts per window.
    annotation_id:
        Identifier of the annotation being analysed.
    role_scope:
        Optional role restriction label for the plot title.
    window_k:
        Window-size parameter used in the analysis.
    output_path:
        Destination path for the PDF figure.
    """

    if not window_counts:
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts_array = np.asarray(list(window_counts), dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 6.0))

    ax.boxplot(
        counts_array,
        positions=[0],
        widths=0.4,
        showfliers=False,
    )

    rng = np.random.default_rng(0)
    jitter = 0.08 * rng.standard_normal(len(counts_array))
    ax.scatter(
        jitter,
        counts_array,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        s=16.0,
    )

    median_value = float(np.median(counts_array))
    ax.axhline(
        median_value,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Median",
    )

    scope_label = role_scope or "any role"
    title = (
        f"{annotation_id} (scope: {scope_label}, K={window_k}, "
        f"windows={len(counts_array)})"
    )
    ax.set_title(title)
    ax.set_ylabel("Positive messages per window")
    ax.set_xticks([0])
    ax.set_xticklabels([annotation_id], rotation=45, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _plot_next_k_histogram(
    counts_next_k: Sequence[int],
    annotation_id: str,
    role_scope: Optional[str],
    window_k: int,
    output_path: Path,
) -> None:
    """Write a histogram of counts in the next K-message window.

    Parameters
    ----------
    counts_next_k:
        Sequence where each entry is the number of later positive messages in
        the next ``window_k`` turns for a single positive occurrence.
    annotation_id:
        Identifier of the annotation being analysed.
    role_scope:
        Optional role restriction label for the plot title.
    window_k:
        Window-size parameter used in the analysis.
    output_path:
        Destination path for the PDF figure.
    """

    if not counts_next_k:
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts_array = np.asarray(list(counts_next_k), dtype=float)
    values, freqs = np.unique(counts_array, return_counts=True)
    total = float(freqs.sum())
    if total <= 0.0:
        return
    probabilities = freqs.astype(float) / total

    fig, ax = plt.subplots(figsize=(4.0, 2))
    ax.bar(
        values,
        probabilities,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    scope_label = role_scope or "any role"
    ax.set_xlabel("Additional positive messages in next K turns")
    ax.set_ylabel("Probability")
    title = (
        f"{annotation_id} (scope: {scope_label}, " f"K={window_k}, trials={int(total)})"
    )
    ax.set_title(title)
    ax.set_xticks(values)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _compute_mean_and_standard_errors(
    window_counts: Sequence[int],
    window_participants: Sequence[str],
) -> Tuple[float, float, float, float, int, int]:
    """Return mean and standard errors for window counts.

    Parameters
    ----------
    window_counts:
        Sequence of positive-message counts per window.
    window_participants:
        Participant ids corresponding to ``window_counts`` entries.

    Returns
    -------
    mean_count:
        Mean number of positive messages per window.
    sd_count:
        Sample standard deviation of window counts.
    se_naive:
        Naive standard error of the mean treating windows as independent.
    se_clustered:
        Participant-clustered standard error of the mean where participants
        are treated as independent clusters.
    n_windows:
        Number of windows contributing to the summary.
    n_participants:
        Number of distinct participants contributing at least one window.
    """

    n_windows = len(window_counts)
    if n_windows == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    total = sum(float(value) for value in window_counts)
    mean_count = total / float(n_windows)

    if n_windows == 1:
        sd_count = 0.0
        se_naive = 0.0
    else:
        variance = 0.0
        for value in window_counts:
            diff = float(value) - mean_count
            variance += diff * diff
        variance /= float(n_windows - 1)
        sd_count = math.sqrt(max(0.0, variance))
        se_naive = sd_count / math.sqrt(float(n_windows))

    # Cluster-robust standard error for the mean treating an intercept-only
    # model with participant-level clustering.
    residuals = [float(value) - mean_count for value in window_counts]
    cluster_sums: Dict[str, float] = {}
    for participant, residual in zip(window_participants, residuals):
        key = str(participant)
        cluster_sums[key] = cluster_sums.get(key, 0.0) + residual

    n_participants = len(cluster_sums)
    if n_participants <= 0:
        return mean_count, sd_count, se_naive, 0.0, n_windows, 0

    numerator = 0.0
    for value in cluster_sums.values():
        numerator += value * value
    variance_cluster = numerator / float(n_windows * n_windows)

    # Apply a small-sample correction analogous to CR1 when multiple clusters
    # are present; fall back to the uncorrected variance when only one
    # participant contributes windows.
    if n_participants > 1:
        variance_cluster *= float(n_participants) / float(n_participants - 1)

    se_clustered = math.sqrt(max(0.0, variance_cluster))
    return mean_count, sd_count, se_naive, se_clustered, n_windows, n_participants


def _run_window_incidence_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    _cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with :func:`run_preprocessed_annotation_job`."""

    message_info, _conversation_messages, annotation_message_positive = family_state

    metadata_by_id_filtered = filter_analysis_metadata(metadata_by_id)
    annotation_id = str(args.annotation_id).strip()

    if not annotation_id:
        print("No annotation id provided; nothing to analyse.")
        return 0

    if annotation_id not in metadata_by_id_filtered:
        print(
            f"Annotation id {annotation_id!r} is not available after "
            "analysis filtering; nothing to analyse.",
        )
        return 0

    window_counts, window_participants = _collect_window_counts_for_annotation(
        annotation_id,
        message_info=message_info,
        annotation_message_positive=annotation_message_positive,
        metadata_by_id=metadata_by_id_filtered,
        window_k=int(args.window_k),
        role_scope=getattr(args, "role_scope", None),
    )

    (
        mean_count,
        sd_count,
        se_naive,
        se_clustered,
        n_windows,
        n_participants,
    ) = _compute_mean_and_standard_errors(window_counts, window_participants)

    if n_windows <= 0:
        print(
            f"No in-scope windows found for annotation {annotation_id!r}; "
            "nothing to summarise.",
        )
        return 0

    role_scope_value = getattr(args, "role_scope", None)
    role_label = role_scope_value or "any"

    print("annotation_id,", annotation_id)
    print("role_scope,", role_label)
    print("window_k,", int(args.window_k))
    print("n_windows,", n_windows)
    print("n_participants,", n_participants)
    print("mean_count,", round3(mean_count))
    print("sd_count,", round3(sd_count))
    print("se_naive,", round3(se_naive))
    print("se_clustered_by_participant,", round3(se_clustered))

    counts_next_k = _collect_next_k_counts_for_annotation(
        annotation_id,
        message_info=message_info,
        annotation_message_positive=annotation_message_positive,
        metadata_by_id=metadata_by_id_filtered,
        window_k=int(args.window_k),
        role_scope=role_scope_value,
    )

    pmf_counts = Counter(counts_next_k)
    total_trials = sum(pmf_counts.values())
    if total_trials > 0:
        print("next_k_count,next_k_probability")
        for count in sorted(pmf_counts.keys()):
            prob = float(pmf_counts[count]) / float(total_trials)
            print(f"{count},{round3(prob)}")

    _plot_next_k_histogram(
        counts_next_k,
        annotation_id=annotation_id,
        role_scope=role_scope_value,
        window_k=int(args.window_k),
        output_path=getattr(args, "output"),
    )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for computing windowed annotation incidence."""

    args = parse_args(argv)
    return run_preprocessed_annotation_job(args, _run_window_incidence_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
