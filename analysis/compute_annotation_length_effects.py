"""Estimate how annotation presence relates to conversation length.

This script fits per-annotation regression models in which transformed
conversation length is the outcome and annotation presence is the predictor.
It operates on the same preprocessed per-message annotation tables used by
``compute_annotation_frequencies.py`` and respects annotation role scopes.

For each annotation id, the script:

* Constructs a message-level dataset restricted to in-scope roles.
* Uses conversation-level message counts as the length variable.
* Fits a linear regression of the form
  length_transform(conversation_length) = beta_0 + beta_1 * I(annotation present).
* Optionally re-computes standard errors with participant-level clustering.

Results are written as a CSV table with one row per annotation id containing
message counts, base rates, and the estimated length effect on the chosen
length scale.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from analysis_utils.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from analysis_utils.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
)
from analysis_utils.csv_utils import write_rows_with_fieldnames
from analysis_utils.effect_summaries import summarise_linear_effect
from analysis_utils.formatting import round3
from analysis_utils.length_cli import add_length_model_arguments, parse_length_args
from analysis_utils.regression_utils import fit_binary_ols_effect
from annotation.io import ParticipantMessageKey
from utils.cli import add_output_path_argument


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the length-effects script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Estimate how annotation presence relates to conversation length "
            "using per-annotation regression models."
        )
    )
    add_length_model_arguments(parser)
    add_output_path_argument(
        parser,
        default_path=Path("analysis/data/annotation_length_effects.csv"),
        help_text=(
            "Output CSV path for per-annotation length-effect estimates. "
            "Each row corresponds to a single annotation id."
        ),
    )
    return parser


def _compute_conversation_lengths(
    conversation_messages: Mapping[ConversationKey, Counter[str]],
) -> Dict[ConversationKey, int]:
    """Return per-conversation message counts.

    Parameters
    ----------
    conversation_messages:
        Mapping from conversation key to a counter of message identifiers
        within that conversation.

    Returns
    -------
    Dict[ConversationKey, int]
        Mapping from conversation key to total message count.
    """

    lengths: Dict[ConversationKey, int] = {}
    for key, counter in conversation_messages.items():
        lengths[key] = sum(counter.values())
    return lengths


def _build_annotation_message_index(
    annotation_message_positive: Mapping[tuple[str, ParticipantMessageKey], bool],
) -> Dict[str, set[ParticipantMessageKey]]:
    """Return per-annotation sets of message keys.

    Parameters
    ----------
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to boolean flags
        indicating whether the annotation is positive on that message.

    Returns
    -------
    Dict[str, set[ParticipantMessageKey]]
        Mapping from annotation id to the set of message keys for which a
        score was observed.
    """

    per_annotation: Dict[str, set[ParticipantMessageKey]] = defaultdict(set)
    for (annotation_id, message_key), _flag in annotation_message_positive.items():
        per_annotation[annotation_id].add(message_key)
    return per_annotation


def _build_regression_sample_for_annotation(
    annotation_id: str,
    *,
    message_info: Mapping[ParticipantMessageKey, tuple[str, ConversationKey]],
    conversation_lengths: Mapping[ConversationKey, int],
    annotation_message_positive: Mapping[tuple[str, ParticipantMessageKey], bool],
    message_keys: Sequence[ParticipantMessageKey],
    metadata_by_id: Mapping[str, AnnotationMetadata],
    length_transform: str,
) -> tuple[list[int], list[float], list[str]]:
    """Return (y, x, clusters) vectors for a single annotation.

    Parameters
    ----------
    annotation_id:
        Identifier of the annotation to analyse.
    message_info:
        Mapping from participant message key to (role, conversation key).
    conversation_lengths:
        Mapping from conversation key to total message count.
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to positivity flags.
    message_keys:
        Sequence of message keys associated with ``annotation_id``.
    metadata_by_id:
        Annotation metadata keyed by annotation id.
    length_transform:
        Name of the transform applied to conversation length as the outcome;
        ``\"log\"`` or ``\"raw\"``.

    Returns
    -------
    y:
        Transformed conversation lengths per message (outcomes).
    x:
        Binary predictors indicating annotation presence per message.
    clusters:
        Participant ids per message for potential clustering.
    """

    meta = metadata_by_id.get(annotation_id)
    if meta is None:
        return [], [], []

    scope = meta.scope

    y: list[float] = []
    x_values: list[float] = []
    clusters: list[str] = []

    for message_key in message_keys:
        info = message_info.get(message_key)
        if info is None:
            continue
        role, conv_key = info
        if not is_role_in_scope(role, scope):
            continue

        length = conversation_lengths.get(conv_key, 0)
        if length <= 0:
            continue

        flag = annotation_message_positive.get((annotation_id, message_key), False)

        if length_transform == "log":
            outcome_value = math.log(float(length))
        else:
            outcome_value = float(length)

        y.append(outcome_value)
        x_values.append(1.0 if flag else 0.0)
        clusters.append(str(message_key[0]))

    return y, x_values, clusters


def _fit_length_effect(
    y: Sequence[float],
    x_values: Sequence[float],
    clusters: Sequence[str],
    *,
    cluster_by_participant: bool,
) -> Optional[tuple[float, float, float, float]]:
    """Return (beta, se, t, p) for the annotation effect on length.

    The function fits a simple linear regression model:

    length = beta_0 + beta_1 * I(annotation present) + error,

    where ``length`` is optionally log-transformed according to the
    ``length_transform`` argument supplied earlier. The returned coefficient
    ``beta`` is the estimated difference in mean transformed length between
    messages with and without the annotation.
    """

    return fit_binary_ols_effect(
        y,
        x_values,
        clusters,
        cluster_by_participant=cluster_by_participant,
    )


def _write_results(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write per-annotation length-effect estimates to ``output_path``."""

    fieldnames = [
        "annotation_id",
        "category",
        "n_messages_scoped",
        "rate_positive",
        "model_type",
        "length_transform",
        "cluster_by_participant",
        "beta_length",
        "se_length",
        "z_value",
        "p_value",
        "odds_ratio",
        "ci_lower",
        "ci_upper",
    ]

    write_rows_with_fieldnames(
        output_path,
        fieldnames,
        rows,
        description="annotation length effects",
    )


def _run_length_effects_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with run_preprocessed_annotation_job for length effects."""

    (
        message_info,
        conversation_messages,
        annotation_message_positive,
    ) = family_state

    # Apply shared analysis filters so that excluded annotations (for example,
    # test-category or synthetic ids) do not participate in length-effect
    # regressions.
    metadata_by_id = filter_analysis_metadata(metadata_by_id)

    if not message_info:
        print("No usable messages discovered in the preprocessed table.")
        return 0

    conversation_lengths = _compute_conversation_lengths(conversation_messages)
    per_annotation_messages = _build_annotation_message_index(
        annotation_message_positive
    )

    # Restrict to annotations that both have LLM score cutoffs and appear in
    # the current table. This mirrors other analysis scripts and avoids
    # synthetic or unused ids present only in metadata.
    annotation_ids = [
        aid
        for aid in metadata_by_id
        if aid in cutoffs_by_id and aid in per_annotation_messages
    ]

    rows: list[dict] = []
    min_messages: int = int(args.min_messages)
    length_transform: str = str(args.length_transform)
    cluster_by_participant: bool = bool(args.cluster_by_participant)

    for annotation_id in annotation_ids:
        message_keys = sorted(per_annotation_messages.get(annotation_id, set()))
        if not message_keys:
            continue

        y, x_values, clusters = _build_regression_sample_for_annotation(
            annotation_id,
            message_info=message_info,
            conversation_lengths=conversation_lengths,
            annotation_message_positive=annotation_message_positive,
            message_keys=message_keys,
            metadata_by_id=metadata_by_id,
            length_transform=length_transform,
        )

        n_messages_scoped = int(len(y))
        n_positive_messages = int(sum(x_values))

        if n_messages_scoped < min_messages:
            continue

        if n_messages_scoped <= 0:
            continue

        rate_positive = round3(float(n_positive_messages) / float(n_messages_scoped))

        estimate = _fit_length_effect(
            y,
            x_values,
            clusters,
            cluster_by_participant=cluster_by_participant,
        )

        meta = metadata_by_id[annotation_id]

        summary = summarise_linear_effect(
            estimate,
            length_transform=length_transform,
        )

        row = {
            "annotation_id": annotation_id,
            "category": meta.category,
            "n_messages_scoped": n_messages_scoped,
            "rate_positive": rate_positive,
            "model_type": "linear",
            "length_transform": length_transform,
            "cluster_by_participant": int(cluster_by_participant),
            "beta_length": summary["beta"],
            "se_length": summary["se"],
            "z_value": summary["z"],
            "p_value": summary["p"],
            "odds_ratio": "",
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
        }
        rows.append(row)

    if not rows:
        print("No annotations met the minimum message threshold for modelling.")
        return 0

    _write_results(Path(args.output), rows)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for estimating annotation length effects."""

    args = parse_length_args(_build_parser, argv)
    return run_preprocessed_annotation_job(args, _run_length_effects_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
