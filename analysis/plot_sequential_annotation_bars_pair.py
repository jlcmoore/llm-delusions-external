"""Plot two per-target sequential dynamics profiles as subplots.

This module provides a small CLI wrapper around
``analysis/plot_sequential_annotation_bars.py`` that renders two
per-target sequential dynamics profiles side by side as subplots in a
single figure. It is intended for cases where two related source
annotations should be compared visually, such as self-harm versus
violence or romantic interest versus sentience.

Both panels share the same window size K, output-prefix for the
sequential dynamics matrices, and effect-source configuration, but can
specify different source and target annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from plot_sequential_annotation_bars import (
    _build_triple_panel_metrics,
    _load_pairwise_window_counts,
    _load_triple_beta_stats,
    _load_triple_window_counts,
    plot_triple_panel_on_axis,
)

from analysis_utils.plot_effects_utils import save_figure
from analysis_utils.sequential_bars_utils import (
    PanelMetrics,
    build_panel_metrics,
    format_annotation_display_label,
    plot_per_target_profile_on_axis,
    print_pairwise_effect_summary,
)
from analysis_utils.sequential_dynamics_cli import parse_window_k_arguments


def _plot_panel_on_axis(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    magnitude_metric: str,
    add_ylabel: bool,
    cond_id: Optional[str],
    triple_targets: Optional[Set[str]],
    pairwise_counts: Optional[Dict[str, Tuple[int, int]]],
    triple_counts: Optional[Dict[str, Tuple[int, int]]],
    triple_beta: Optional[Dict[str, Tuple[float, float, float]]],
    show_effect_annotations: bool,
) -> Tuple[
    Optional[plt.Container],
    Optional[plt.Container],
    Optional[plt.Container],
]:
    """Render a per-target profile onto an existing Matplotlib axis.

    Returns the baseline, pairwise-conditional, and triple-conditional
    artists (the latter is None in pairwise-only mode) so that the
    caller can construct a shared legend across subplots.
    """

    if cond_id and triple_targets and pairwise_counts is not None and triple_counts:
        (
            baseline_artist,
            conditional_artist_pairwise,
            conditional_artist_triple,
        ) = plot_triple_panel_on_axis(
            axis,
            panel=panel,
            k=k,
            effect_source=effect_source,
            magnitude_metric=magnitude_metric,
            cond_id=cond_id,
            triple_targets=triple_targets,
            pairwise_counts=pairwise_counts,
            triple_counts=triple_counts,
            triple_beta=triple_beta,
            show_effect_annotations=show_effect_annotations,
        )
        if add_ylabel:
            if effect_source == "beta":
                axis.set_ylabel(
                    f"P(occurs >= 1x in {k} msgs.)",
                    fontsize=9,
                )
            else:
                axis.set_ylabel(
                    "Per-message rate of Y (per-step within K)",
                    fontsize=9,
                )
        return baseline_artist, conditional_artist_pairwise, conditional_artist_triple

    baseline_artist, conditional_artist = plot_per_target_profile_on_axis(
        axis,
        panel=panel,
        k=k,
        effect_source=effect_source,
        magnitude_metric=magnitude_metric,
        add_ylabel=add_ylabel,
        add_arrows=True,
        show_effect_labels=show_effect_annotations,
    )
    return baseline_artist, conditional_artist, None


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the paired per-target plot."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot two per-target sequential annotation profiles side by side "
            "as subplots from precomputed X->Y matrix CSV tables."
        )
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("analysis") / "data" / "sequential_dynamics" / "base",
        help=(
            "Prefix of sequential dynamics CSV tables produced by "
            "compute_sequential_annotation_dynamics.py. The single-K matrix "
            "file is expected at '<prefix>_K{K}_matrix.csv', for example "
            "'analysis/data/sequential_dynamics/base_K5_matrix.csv'."
        ),
    )
    parser.add_argument(
        "--window-k",
        type=int,
        action="append",
        help=(
            "Window size K in messages used when computing sequential "
            "dynamics. Exactly one K must be provided."
        ),
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("analysis")
        / "figures"
        / "sequential_enrichment_profile_pair_K{K}.pdf",
        help=(
            "Destination PDF path for the paired per-target plot. The "
            "placeholder '{K}' in the default will be replaced with the "
            "selected window size."
        ),
    )
    parser.add_argument(
        "--effect-source",
        choices=["beta", "enrichment"],
        default="beta",
        help=(
            "Effect-size source for the y-axis: 'beta' uses the K-window "
            "occurrence probabilities and Beta-model uncertainty, while "
            "'enrichment' uses per-step per-message rates with approximate "
            "Beta intervals."
        ),
    )
    parser.add_argument(
        "--order-by-effect-size",
        action="store_true",
        help=(
            "Order targets in each panel by the absolute difference between "
            "the global baseline and the conditional rate for the selected "
            "source. When omitted, targets appear in the order provided via "
            "--left-target-id and --right-target-id respectively, or "
            "alphabetically when no explicit order is given."
        ),
    )
    parser.add_argument(
        "--left-source-id",
        type=str,
        required=True,
        help="Source annotation id for the left subplot.",
    )
    parser.add_argument(
        "--left-target-id",
        type=str,
        action="append",
        help=(
            "Target annotation id for the left subplot. May be provided "
            "multiple times; when omitted, all annotations present as targets "
            "for the selected left source are shown."
        ),
    )
    parser.add_argument(
        "--left-cond-id",
        type=str,
        help=(
            "Optional conditioning annotation id Y for the left subplot. "
            "When provided, the left panel uses the co-window triples table "
            "and treats --left-conditional-target-id values as Z annotations "
            "for which P(Z | X, Y-in-window) is plotted alongside P(Z | X)."
        ),
    )
    parser.add_argument(
        "--left-conditional-target-id",
        type=str,
        action="append",
        help=(
            "Third annotation id Z for the left subplot whose conditional "
            "probability P(Z | X, Y-in-window) should be plotted. When "
            "omitted, any overlapping --left-target-id values that appear in "
            "the triples table are treated as conditional targets."
        ),
    )
    parser.add_argument(
        "--right-source-id",
        type=str,
        required=True,
        help="Source annotation id for the right subplot.",
    )
    parser.add_argument(
        "--right-target-id",
        type=str,
        action="append",
        help=(
            "Target annotation id for the right subplot. May be provided "
            "multiple times; when omitted, all annotations present as targets "
            "for the selected right source are shown."
        ),
    )
    parser.add_argument(
        "--right-cond-id",
        type=str,
        help=(
            "Optional conditioning annotation id Y for the right subplot. "
            "When provided, the right panel uses the co-window triples table "
            "and treats --right-conditional-target-id values as Z annotations "
            "for which P(Z | X, Y-in-window) is plotted alongside P(Z | X)."
        ),
    )
    parser.add_argument(
        "--right-conditional-target-id",
        type=str,
        action="append",
        help=(
            "Third annotation id Z for the right subplot whose conditional "
            "probability P(Z | X, Y-in-window) should be plotted. When "
            "omitted, any overlapping --right-target-id values that appear in "
            "the triples table are treated as conditional targets."
        ),
    )
    parser.add_argument(
        "--magnitude-metric",
        choices=["odds", "risk"],
        default="odds",
        help=(
            "Metric used for the per-target change annotation in each panel: "
            "'odds' uses the odds ratio between conditional and baseline "
            "probabilities, while 'risk' uses the risk ratio "
            "(conditional probability divided by baseline probability)."
        ),
    )
    parser.add_argument(
        "--hide-effect-annotations",
        action="store_true",
        help=(
            "When set, omit odds/risk ratio annotations and arrows from the "
            "per-target plots in both panels. Printed summary tables are "
            "unaffected."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the paired per-target plotting script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    args.window_k = parse_window_k_arguments(parser, args.window_k)
    if len(args.window_k) != 1:
        parser.error("Exactly one --window-k value must be provided.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for paired per-target sequential-dynamics plots."""

    plt.switch_backend("Agg")

    args = parse_args(argv)
    k: int = int(args.window_k[0])

    left_cond_id: Optional[str] = getattr(args, "left_cond_id", None)
    right_cond_id: Optional[str] = getattr(args, "right_cond_id", None)

    left_triple_targets: Optional[Set[str]] = None
    right_triple_targets: Optional[Set[str]] = None

    try:
        if left_cond_id:
            left_panel, left_triple_targets = _build_triple_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id=args.left_source_id,
                cond_id=left_cond_id,
                pairwise_targets_raw=args.left_target_id,
                conditional_targets_raw=getattr(
                    args,
                    "left_conditional_target_id",
                    None,
                ),
                order_by_effect_size=bool(args.order_by_effect_size),
            )
        else:
            left_panel = build_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id_raw=args.left_source_id,
                target_ids_raw=args.left_target_id,
                effect_source=args.effect_source,
                order_by_effect_size=bool(args.order_by_effect_size),
            )

        if right_cond_id:
            right_panel, right_triple_targets = _build_triple_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id=args.right_source_id,
                cond_id=right_cond_id,
                pairwise_targets_raw=args.right_target_id,
                conditional_targets_raw=getattr(
                    args,
                    "right_conditional_target_id",
                    None,
                ),
                order_by_effect_size=bool(args.order_by_effect_size),
            )
        else:
            right_panel = build_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id_raw=args.right_source_id,
                target_ids_raw=args.right_target_id,
                effect_source=args.effect_source,
                order_by_effect_size=bool(args.order_by_effect_size),
            )
    except ValueError as exc:
        print(str(exc))
        return 2

    if not left_panel.targets or not right_panel.targets:
        print("At least one panel had no targets; nothing to plot.")
        return 1

    if left_cond_id and left_triple_targets:
        matrix_path_left = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_matrix.csv",
        )
        left_pairwise_counts = _load_pairwise_window_counts(
            matrix_path_left,
            k,
            left_panel.source_id,
            left_panel.targets,
        )
        triples_path_left = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_triples_cowindow.csv",
        )
        left_triple_counts = _load_triple_window_counts(
            triples_path_left,
            k,
            left_panel.source_id,
            left_cond_id,
            left_triple_targets,
        )
        for target in left_panel.targets:
            base_p = float(left_panel.baseline_means[target])
            pair_trials, pair_successes = left_pairwise_counts.get(target, (0, 0))
            p_x = float(pair_successes) / float(pair_trials) if pair_trials > 0 else 0.0

            delta_pair = p_x - base_p
            if base_p > 0.0:
                rr_pair = p_x / base_p
            else:
                rr_pair = float("nan")
            if 0.0 < base_p < 1.0 and 0.0 < p_x < 1.0 and base_p != 1.0:
                base_odds = base_p / (1.0 - base_p)
                cond_odds = p_x / (1.0 - p_x)
                or_pair = cond_odds / base_odds if base_odds > 0.0 else float("nan")
            else:
                or_pair = float("nan")
            print(
                f"{target:40s} {'pair':>8s} "
                f"{base_p:10.3f} {p_x:12.3f} "
                f"{delta_pair:10.3f} {rr_pair:10.3f} {or_pair:10.3f}"
            )

            if target not in left_triple_targets:
                continue
            triple_trials, triple_successes = left_triple_counts.get(target, (0, 0))
            if triple_trials <= 0:
                continue
            p_xy = float(triple_successes) / float(triple_trials)
            delta_triple = p_xy - p_x
            if p_x > 0.0:
                rr_triple = p_xy / p_x
            else:
                rr_triple = float("nan")
            if 0.0 < p_x < 1.0 and 0.0 < p_xy < 1.0 and p_x != 1.0:
                base_odds = p_x / (1.0 - p_x)
                cond_odds = p_xy / (1.0 - p_xy)
                or_triple = cond_odds / base_odds if base_odds > 0.0 else float("nan")
            else:
                or_triple = float("nan")
            print(
                f"{target:40s} {'triple':>8s} "
                f"{p_x:10.3f} {p_xy:12.3f} "
                f"{delta_triple:10.3f} {rr_triple:10.3f} {or_triple:10.3f}"
            )
    else:
        print_pairwise_effect_summary(left_panel, panel_label="left panel")

    if right_cond_id and right_triple_targets:
        matrix_path_right = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_matrix.csv",
        )
        right_pairwise_counts = _load_pairwise_window_counts(
            matrix_path_right,
            k,
            right_panel.source_id,
            right_panel.targets,
        )
        triples_path_right = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_triples_cowindow.csv",
        )
        right_triple_counts = _load_triple_window_counts(
            triples_path_right,
            k,
            right_panel.source_id,
            right_cond_id,
            right_triple_targets,
        )
        for target in right_panel.targets:
            base_p = float(right_panel.baseline_means[target])
            pair_trials, pair_successes = right_pairwise_counts.get(target, (0, 0))
            p_x = float(pair_successes) / float(pair_trials) if pair_trials > 0 else 0.0

            delta_pair = p_x - base_p
            if base_p > 0.0:
                rr_pair = p_x / base_p
            else:
                rr_pair = float("nan")
            if 0.0 < base_p < 1.0 and 0.0 < p_x < 1.0 and base_p != 1.0:
                base_odds = base_p / (1.0 - base_p)
                cond_odds = p_x / (1.0 - p_x)
                or_pair = cond_odds / base_odds if base_odds > 0.0 else float("nan")
            else:
                or_pair = float("nan")
            print(
                f"{target:40s} {'pair':>8s} "
                f"{base_p:10.3f} {p_x:12.3f} "
                f"{delta_pair:10.3f} {rr_pair:10.3f} {or_pair:10.3f}"
            )

            if target not in right_triple_targets:
                continue
            triple_trials, triple_successes = right_triple_counts.get(target, (0, 0))
            if triple_trials <= 0:
                continue
            p_xy = float(triple_successes) / float(triple_trials)
            delta_triple = p_xy - p_x
            if p_x > 0.0:
                rr_triple = p_xy / p_x
            else:
                rr_triple = float("nan")
            if 0.0 < p_x < 1.0 and 0.0 < p_xy < 1.0 and p_x != 1.0:
                base_odds = p_x / (1.0 - p_x)
                cond_odds = p_xy / (1.0 - p_xy)
                or_triple = cond_odds / base_odds if base_odds > 0.0 else float("nan")
            else:
                or_triple = float("nan")
            print(
                f"{target:40s} {'triple':>8s} "
                f"{p_x:10.3f} {p_xy:12.3f} "
                f"{delta_triple:10.3f} {rr_triple:10.3f} {or_triple:10.3f}"
            )
    else:
        print_pairwise_effect_summary(
            right_panel,
            panel_label="right panel",
            leading_newline=True,
        )

    left_count = max(len(left_panel.targets), 1)
    right_count = max(len(right_panel.targets), 1)
    base_width = 0.5
    per_target_width = 0.9
    left_width = base_width + per_target_width * float(left_count)
    right_width = base_width + per_target_width * float(right_count)
    figure_width = left_width + right_width
    figure_height = 2.5

    figure, (axis_left, axis_right) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        sharey=False,
        gridspec_kw={"width_ratios": [left_width, right_width]},
    )

    # Preload counts and Beta stats for plotting when conditional
    # triples are requested.
    left_pairwise_counts: Optional[Dict[str, Tuple[int, int]]] = None
    left_triple_counts: Optional[Dict[str, Tuple[int, int]]] = None
    left_triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None
    if left_cond_id and left_triple_targets:
        matrix_path_left = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_matrix.csv",
        )
        left_pairwise_counts = _load_pairwise_window_counts(
            matrix_path_left,
            k,
            left_panel.source_id,
            left_panel.targets,
        )
        triples_path_left = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_triples_cowindow.csv",
        )
        left_triple_counts = _load_triple_window_counts(
            triples_path_left,
            k,
            left_panel.source_id,
            left_cond_id,
            left_triple_targets,
        )
        if args.effect_source == "beta":
            left_triple_beta = _load_triple_beta_stats(
                triples_path_left,
                k,
                left_panel.source_id,
                left_cond_id,
                left_triple_targets,
            )

    right_pairwise_counts: Optional[Dict[str, Tuple[int, int]]] = None
    right_triple_counts: Optional[Dict[str, Tuple[int, int]]] = None
    right_triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None
    if right_cond_id and right_triple_targets:
        matrix_path_right = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_matrix.csv",
        )
        right_pairwise_counts = _load_pairwise_window_counts(
            matrix_path_right,
            k,
            right_panel.source_id,
            right_panel.targets,
        )
        triples_path_right = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_triples_cowindow.csv",
        )
        right_triple_counts = _load_triple_window_counts(
            triples_path_right,
            k,
            right_panel.source_id,
            right_cond_id,
            right_triple_targets,
        )
        if args.effect_source == "beta":
            right_triple_beta = _load_triple_beta_stats(
                triples_path_right,
                k,
                right_panel.source_id,
                right_cond_id,
                right_triple_targets,
            )

    (
        left_baseline_artist,
        left_conditional_artist,
        left_triple_artist,
    ) = _plot_panel_on_axis(
        axis_left,
        panel=left_panel,
        k=k,
        effect_source=args.effect_source,
        magnitude_metric=args.magnitude_metric,
        add_ylabel=True,
        cond_id=left_cond_id,
        triple_targets=left_triple_targets,
        pairwise_counts=left_pairwise_counts,
        triple_counts=left_triple_counts,
        triple_beta=left_triple_beta,
        show_effect_annotations=not bool(
            getattr(args, "hide_effect_annotations", False)
        ),
    )
    (
        _,
        right_conditional_artist,
        right_triple_artist,
    ) = _plot_panel_on_axis(
        axis_right,
        panel=right_panel,
        k=k,
        effect_source=args.effect_source,
        magnitude_metric=args.magnitude_metric,
        add_ylabel=False,
        cond_id=right_cond_id,
        triple_targets=right_triple_targets,
        pairwise_counts=right_pairwise_counts,
        triple_counts=right_triple_counts,
        triple_beta=right_triple_beta,
        show_effect_annotations=not bool(
            getattr(args, "hide_effect_annotations", False)
        ),
    )

    axis_left.set_title(
        f"Source: {format_annotation_display_label(left_panel.source_id)}",
        fontsize=9,
    )
    axis_right.set_title(
        f"Source: {format_annotation_display_label(right_panel.source_id)}",
        fontsize=9,
    )

    legend_handles = []
    legend_labels = []
    seen_labels: Set[str] = set()

    if left_baseline_artist is not None:
        label = "Global baseline"
        if label not in seen_labels:
            legend_handles.append(left_baseline_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if left_conditional_artist is not None:
        label = f"Following {format_annotation_display_label(left_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(left_conditional_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if right_conditional_artist is not None:
        label = f"Following {format_annotation_display_label(right_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(right_conditional_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if left_triple_artist is not None and left_cond_id:
        label = (
            f"Following {format_annotation_display_label(left_panel.source_id)} and "
            f"{format_annotation_display_label(left_cond_id)} in window"
        )
        if label not in seen_labels:
            legend_handles.append(left_triple_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if right_triple_artist is not None and right_cond_id:
        label = (
            f"Following {format_annotation_display_label(right_panel.source_id)} and "
            f"{format_annotation_display_label(right_cond_id)} in window"
        )
        if label not in seen_labels:
            legend_handles.append(right_triple_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if legend_handles:
        label_count = len(legend_labels)
        if label_count <= 3:
            ncol = label_count
            legend_kwargs = {"loc": "lower center"}
        else:
            # Use three columns so the legend wraps to two rows, and
            # move it slightly below the figure to keep the panels
            # unobstructed.
            ncol = 2
            legend_kwargs = {
                "loc": "upper center",
                "bbox_to_anchor": (0.5, 0.18),
            }
        figure.legend(
            handles=legend_handles,
            labels=legend_labels,
            ncol=ncol,
            fontsize=8,
            **legend_kwargs,
        )

    figure.tight_layout()

    if label_count > 3:
        figure.subplots_adjust(
            left=0.08, right=0.99, bottom=0.45, top=0.89, wspace=0.15
        )

    figure_path = args.figure_path
    if "{K}" in str(figure_path):
        figure_path = Path(str(figure_path).format(K=k))
    save_figure(figure_path, figure)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
