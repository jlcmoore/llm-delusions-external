"""Plot per-target sequential dynamics for a single source annotation.

This module consumes a single-K X->Y matrix CSV written by
``compute_sequential_annotation_dynamics.py`` and produces a bar-style plot
for one source annotation X and a selected list of target annotations Y. For
each target, it compares:

* A global baseline rate for Y (either per-message or K-window occurrence).
* The corresponding rate for Y within K messages after X.

Both effect sources used in the heatmap script are supported:

* ``beta``: K-window occurrence probabilities with Beta-model uncertainty.
* ``enrichment``: per-step per-message rates and approximate Beta intervals.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils.sequential_bars_utils import (
    PanelMetrics,
    annotation_color_for_label,
    build_panel_metrics,
    compute_effect_metrics,
    format_annotation_display_label,
    plot_per_target_profile_on_axis,
)
from analysis_utils.sequential_dynamics_cli import parse_window_k_arguments


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the per-target sequential-dynamics plot."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot per-target sequential annotation dynamics for a single "
            "source annotation from a precomputed X->Y matrix CSV."
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
        default=Path("analysis") / "figures" / "sequential_enrichment_profile_K{K}.pdf",
        help=(
            "Destination PDF path for the per-target plot. The placeholder "
            "'{K}' in the default will be replaced with the selected window "
            "size."
        ),
    )
    parser.add_argument(
        "--source-id",
        type=str,
        required=True,
        help="Source annotation id X whose conditional effects are plotted.",
    )
    parser.add_argument(
        "--cond-id",
        type=str,
        help=(
            "Optional conditioning annotation id Y. When provided, the plot "
            "uses the co-window triples table and treats --target-id values "
            "as third annotations Z, comparing P(Z | X) to "
            "P(Z | X, Y-in-window). When omitted, the plot uses the X->Y "
            "pairwise matrix as before."
        ),
    )
    parser.add_argument(
        "--target-id",
        type=str,
        action="append",
        help=(
            "Target annotation id. In the default pairwise mode, these are "
            "Y annotations to include from the X->Y matrix. When --cond-id "
            "is set and --conditional-target-id is omitted, these are Z "
            "annotations drawn from the co-window triples table for the "
            "selected (X, Y, K). May be provided multiple times; when "
            "omitted, all available targets are shown."
        ),
    )
    parser.add_argument(
        "--conditional-target-id",
        type=str,
        action="append",
        help=(
            "Annotation id Z whose conditional probability P(Z | X, "
            "Y-in-window) should be plotted alongside P(Z | X). These are "
            "drawn from the co-window triples table. When omitted, all "
            "Z annotations observed in the triples table (or those provided "
            "via --target-id when --cond-id is set) are treated as "
            "conditional targets."
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
        "--magnitude-metric",
        choices=["odds", "risk"],
        default="odds",
        help=(
            "Metric used for the per-target change annotation: 'odds' uses the "
            "odds ratio between conditional and baseline probabilities, while "
            "'risk' uses the risk ratio (conditional probability divided by "
            "baseline probability)."
        ),
    )
    parser.add_argument(
        "--hide-effect-annotations",
        action="store_true",
        help=(
            "When set, omit odds/risk ratio annotations and arrows from the "
            "per-target plot. Printed summary tables are unaffected."
        ),
    )
    parser.add_argument(
        "--order-by-effect-size",
        action="store_true",
        help=(
            "Order targets by the absolute difference between the global "
            "baseline and the conditional rate for the selected source. "
            "When omitted, targets appear in the order provided via "
            "--target-id, or alphabetically when no explicit order is given."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the per-target plotting script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    args.window_k = parse_window_k_arguments(parser, args.window_k)
    if len(args.window_k) != 1:
        parser.error("Exactly one --window-k value must be provided.")
    return args


def _build_triple_panel_metrics(
    *,
    output_prefix: Path,
    k: int,
    source_id: str,
    cond_id: str,
    pairwise_targets_raw: Optional[Sequence[str]],
    conditional_targets_raw: Optional[Sequence[str]],
    order_by_effect_size: bool,
) -> Tuple[PanelMetrics, Set[str]]:
    """Return PanelMetrics for a co-window triple configuration.

    This helper loads X,Y,Z co-window statistics from the triples CSV
    produced by ``compute_sequential_annotation_dynamics.py`` and adapts
    them to the PanelMetrics interface used by the standard bar plots.
    and P(Z | X, Y-in-window) for one or more Z. The final panel always
    includes the conditioning Y row so that P(Y | X) can be visualised
    alongside the conditional Z effects.
    """

    triples_path = output_prefix.with_name(
        f"{output_prefix.name}_K{k}_triples_cowindow.csv",
    )
    triples_path = triples_path.expanduser().resolve()
    if not triples_path.is_file():
        raise ValueError(f"Triples CSV not found at {triples_path}")

    triple_baseline: Dict[str, float] = {}
    triple_conditional: Dict[str, float] = {}

    with triples_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_k = int(row.get("K", "0"))
            except ValueError:
                continue
            if row_k != k:
                continue
            if row.get("X") != source_id or row.get("Y") != cond_id:
                continue
            z_value = row.get("Z")
            if not z_value:
                continue

            try:
                base_p = float(row.get("p_z_within_K_given_X", "0"))
            except (TypeError, ValueError):
                base_p = 0.0
            try:
                cond_p = float(row.get("p_z_within_K_given_XY", "0"))
            except (TypeError, ValueError):
                cond_p = 0.0

            triple_baseline[z_value] = base_p
            triple_conditional[z_value] = cond_p

    if not triple_baseline:
        raise ValueError(
            "No co-window triples were found for the requested "
            f"K={k}, X={source_id!r}, Y={cond_id!r}.",
        )

    # Determine which Z ids to retain from the triples table. When explicit
    # conditional targets are provided, use them; otherwise, fall back to
    # any --target-id values or finally to all observed Z ids. The
    # conditioning annotation Y itself is always treated as a pairwise
    # target only, so it is excluded from the Z set even if requested.
    z_ids: List[str]
    if conditional_targets_raw:
        z_ids = [
            z for z in conditional_targets_raw if z in triple_baseline and z != cond_id
        ]
    elif pairwise_targets_raw:
        z_ids = [
            z for z in pairwise_targets_raw if z in triple_baseline and z != cond_id
        ]
    else:
        z_ids = sorted(z for z in triple_baseline if z != cond_id)

    if not z_ids:
        raise ValueError(
            "No Z annotations remained after filtering by --target-id; "
            "nothing to plot.",
        )

    # Determine pairwise-only targets: any --target-id not used as a
    # conditional Z and not equal to the conditioning Y.
    pairwise_only: List[str] = []
    if pairwise_targets_raw:
        for target in pairwise_targets_raw:
            if target == cond_id:
                continue
            if target in z_ids:
                continue
            if target not in pairwise_only:
                pairwise_only.append(target)

    # Build a matrix-based panel that covers the conditioning Y, all Z ids,
    # and any additional pairwise-only targets so that P(Y | X) and P(T | X)
    # use the standard logic.
    matrix_target_ids: List[str] = [cond_id]
    for z in z_ids:
        if z not in matrix_target_ids:
            matrix_target_ids.append(z)
    for target in pairwise_only:
        if target not in matrix_target_ids:
            matrix_target_ids.append(target)

    base_panel = build_panel_metrics(
        output_prefix=output_prefix,
        k=k,
        source_id_raw=source_id,
        target_ids_raw=matrix_target_ids,
        effect_source="beta",
        order_by_effect_size=False,
    )

    baseline_means: Dict[str, float] = dict(base_panel.baseline_means)
    baseline_cis: Dict[str, Tuple[float, float]] = dict(base_panel.baseline_cis)
    conditional_means: Dict[str, float] = dict(base_panel.conditional_means)
    conditional_cis: Dict[str, Tuple[float, float]] = dict(
        base_panel.conditional_cis,
    )

    # Final target ordering: Y first, followed by pairwise-only targets and
    # then Z ids. When requested, order Z by absolute change in conditional
    # probability.
    if order_by_effect_size:
        z_ids = sorted(
            z_ids,
            key=lambda name: abs(
                float(conditional_means.get(name, 0.0))
                - float(baseline_means.get(name, 0.0))
            ),
            reverse=True,
        )
    targets: List[str] = [cond_id]
    targets.extend(pairwise_only)
    targets.extend([z for z in z_ids if z != cond_id])

    panel = PanelMetrics(
        source_id=base_panel.source_id,
        targets=targets,
        baseline_means=baseline_means,
        baseline_cis=baseline_cis,
        conditional_means=conditional_means,
        conditional_cis=conditional_cis,
    )
    return panel, set(z_ids)


def _load_pairwise_window_counts(
    matrix_path: Path,
    k: int,
    source_id: str,
    target_ids: Sequence[str],
) -> Dict[str, Tuple[int, int]]:
    """Return per-target (trials, successes) for X->target K-windows."""

    resolved = matrix_path.expanduser().resolve()
    if not resolved.is_file():
        return {}

    wanted = set(target_ids)
    counts: Dict[str, Tuple[int, int]] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_k = int(row.get("K", "0"))
            except ValueError:
                continue
            if row_k != k:
                continue
            if row.get("X") != source_id:
                continue
            target = row.get("Y")
            if target not in wanted:
                continue
            try:
                trials = int(row.get("beta_trials", "0"))
            except ValueError:
                trials = 0
            try:
                successes = int(row.get("beta_successes", "0"))
            except ValueError:
                successes = 0
            counts[target] = (trials, successes)

    return counts


def _load_triple_window_counts(
    triples_path: Path,
    k: int,
    source_id: str,
    cond_id: str,
    z_ids: Set[str],
) -> Dict[str, Tuple[int, int]]:
    """Return per-Z (trials_XY, successes_XYZ) for X,Y->Z K-windows."""

    resolved = triples_path.expanduser().resolve()
    if not resolved.is_file():
        return {}

    counts: Dict[str, Tuple[int, int]] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_k = int(row.get("K", "0"))
            except ValueError:
                continue
            if row_k != k:
                continue
            if row.get("X") != source_id or row.get("Y") != cond_id:
                continue
            z_value = row.get("Z")
            if z_value not in z_ids:
                continue
            try:
                trials_xy = int(row.get("trials_XY", "0"))
            except ValueError:
                trials_xy = 0
            try:
                successes_xyz = int(row.get("successes_XYZ", "0"))
            except ValueError:
                successes_xyz = 0
            counts[z_value] = (trials_xy, successes_xyz)

    return counts


def _load_triple_beta_stats(
    triples_path: Path,
    k: int,
    source_id: str,
    cond_id: str,
    z_ids: Set[str],
) -> Dict[str, Tuple[float, float, float]]:
    """Return per-Z (mean, ci_low, ci_high) for X,Y->Z Beta posteriors."""

    resolved = triples_path.expanduser().resolve()
    if not resolved.is_file():
        return {}

    stats: Dict[str, Tuple[float, float, float]] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_k = int(row.get("K", "0"))
            except ValueError:
                continue
            if row_k != k:
                continue
            if row.get("X") != source_id or row.get("Y") != cond_id:
                continue
            z_value = row.get("Z")
            if z_value not in z_ids:
                continue
            try:
                mean = float(row.get("beta_p_window_given_XY", "0"))
            except (TypeError, ValueError):
                mean = 0.0
            try:
                ci_low = float(row.get("beta_p_window_ci_low", "0"))
            except (TypeError, ValueError):
                ci_low = 0.0
            try:
                ci_high = float(row.get("beta_p_window_ci_high", "0"))
            except (TypeError, ValueError):
                ci_high = 0.0
            stats[z_value] = (mean, ci_low, ci_high)

    return stats


def _annotate_or_step(
    axis: plt.Axes,
    *,
    x_position: float,
    y_start: float,
    y_end: float,
    magnitude: float,
    horizontal_alignment: str,
    text_color: str,
    show_text: bool = True,
) -> None:
    """Draw an odds- or risk-ratio arrow and label between two points."""

    midpoint_y = (y_start + y_end) / 2.0
    axis.annotate(
        "",
        xy=(x_position, y_end),
        xytext=(x_position, y_start),
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "lw": 0.8,
        },
        zorder=2,
    )
    if show_text:
        axis.text(
            x_position + (0.05 if horizontal_alignment == "left" else -0.05),
            midpoint_y,
            f"{magnitude:.1f}x",
            fontsize=7,
            ha=horizontal_alignment,
            va="center",
            color=text_color,
        )


def plot_triple_panel_on_axis(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    magnitude_metric: str,
    cond_id: str,
    triple_targets: Set[str],
    pairwise_counts: Dict[str, Tuple[int, int]],
    triple_counts: Dict[str, Tuple[int, int]],
    triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None,
    show_effect_annotations: bool = True,
) -> Tuple[plt.Container, plt.Container, plt.Container]:
    """Render a triple-conditional profile onto an existing axis."""

    baseline_artist, conditional_artist_pairwise = plot_per_target_profile_on_axis(
        axis,
        panel=panel,
        k=k,
        effect_source=effect_source,
        magnitude_metric=magnitude_metric,
        add_arrows=False,
        add_ylabel=False,
    )

    x_positions = np.arange(len(panel.targets), dtype=float)

    # Overlay triple-conditioned points in the conditioning label colour
    # at P(Z | X, Y-in-window) for Z in triple_targets. When Beta triple
    # statistics are available and effect_source='beta', use the
    # posterior means and intervals; otherwise fall back to MLEs.
    conditional_y = np.full(len(panel.targets), np.nan, dtype=float)
    conditional_yerr = np.zeros((2, len(panel.targets)), dtype=float)
    use_beta = effect_source == "beta" and triple_beta is not None
    for idx, target in enumerate(panel.targets):
        if target not in triple_targets:
            continue
        if use_beta and target in triple_beta:
            mean, ci_low, ci_high = triple_beta[target]
            conditional_y[idx] = mean
            conditional_yerr[0, idx] = mean - ci_low
            conditional_yerr[1, idx] = ci_high - mean
        else:
            triple_trials, triple_successes = triple_counts.get(target, (0, 0))
            if triple_trials <= 0:
                continue
            conditional_y[idx] = float(triple_successes) / float(triple_trials)
    cond_color = annotation_color_for_label(cond_id)
    triple_indices = [
        idx
        for idx, target in enumerate(panel.targets)
        if target in triple_targets and np.isfinite(conditional_y[idx])
    ]
    if use_beta:
        conditional_artist_triple = axis.errorbar(
            x_positions[triple_indices],
            conditional_y[triple_indices],
            yerr=conditional_yerr[:, triple_indices],
            fmt="o",
            color=cond_color,
            ecolor=cond_color,
            elinewidth=1.0,
            capsize=3.0,
            zorder=5,
        )
    else:
        conditional_artist_triple = axis.errorbar(
            x_positions[triple_indices],
            conditional_y[triple_indices],
            fmt="o",
            color=cond_color,
            ecolor=cond_color,
            elinewidth=1.0,
            capsize=3.0,
            zorder=5,
        )

    # Add two-step odds-ratio annotations for triple targets:
    # P(Z) -> P(Z|X) and P(Z|X) -> P(Z|X,Y).
    if show_effect_annotations:
        for idx, target in enumerate(panel.targets):
            base_p = float(panel.baseline_means[target])
            pair_trials, pair_successes = pairwise_counts.get(target, (0, 0))
            if pair_trials <= 0:
                continue
            p_x = float(pair_successes) / float(pair_trials)

            # First step: P(Z) -> P(Z|X).
            if 0.0 < base_p < 1.0 and 0.0 < p_x < 1.0 and base_p != 1.0:
                base_odds = base_p / (1.0 - base_p)
                cond_odds = p_x / (1.0 - p_x)
                if base_odds > 0.0:
                    or_global = cond_odds / base_odds
                    _annotate_or_step(
                        axis,
                        x_position=float(x_positions[idx]),
                        y_start=base_p,
                        y_end=p_x,
                        magnitude=or_global,
                        horizontal_alignment="left",
                        text_color=annotation_color_for_label(panel.source_id),
                        show_text=show_effect_annotations,
                    )

            # Second step: P(Z|X) -> P(Z|X,Y) for triple targets only.
            if target not in triple_targets:
                continue
            triple_trials, triple_successes = triple_counts.get(target, (0, 0))
            if triple_trials <= 0:
                continue
            p_xy = float(triple_successes) / float(triple_trials)
            if 0.0 < p_x < 1.0 and 0.0 < p_xy < 1.0 and p_x != 1.0:
                base_odds = p_x / (1.0 - p_x)
                cond_odds = p_xy / (1.0 - p_xy)
                if base_odds > 0.0:
                    or_cond = cond_odds / base_odds
                    _annotate_or_step(
                        axis,
                        x_position=float(x_positions[idx]),
                        y_start=p_x,
                        y_end=p_xy,
                        magnitude=or_cond,
                        horizontal_alignment="right",
                        text_color=cond_color,
                        show_text=show_effect_annotations,
                    )

    return baseline_artist, conditional_artist_pairwise, conditional_artist_triple


def _plot_per_target_bars(
    output_path: Path,
    k: int,
    panel: PanelMetrics,
    effect_source: str,
    magnitude_metric: str,
    show_effect_annotations: bool,
    cond_id: Optional[str] = None,
    triple_targets: Optional[Set[str]] = None,
    pairwise_counts: Optional[Dict[str, Tuple[int, int]]] = None,
    triple_counts: Optional[Dict[str, Tuple[int, int]]] = None,
    triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> None:
    """Render and save a per-target bar-style plot for a single source."""

    if not panel.targets:
        print("No targets were selected for the per-target plot; nothing to do.")
        return

    output_path = output_path.expanduser().resolve()
    if "{K}" in str(output_path):
        output_path = Path(str(output_path).format(K=k))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(2 + float(len(panel.targets)), 2.5))

    if cond_id and triple_targets and pairwise_counts is not None and triple_counts:
        source_label = format_annotation_display_label(panel.source_id)
        cond_label = format_annotation_display_label(cond_id)
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

        if effect_source == "beta":
            axis.set_ylabel(f"P(occurs >= 1x in {k} msgs.)", fontsize=9)
        else:
            axis.set_ylabel(
                "Per-message rate of Y (per-step within K)",
                fontsize=9,
            )

        handles = [
            baseline_artist,
            conditional_artist_pairwise,
            conditional_artist_triple,
        ]
        labels = [
            "Global baseline",
            f"Following {source_label}",
            f"Following {source_label} and {cond_label} in window",
        ]
        axis.legend(handles=handles, labels=labels, fontsize=8)
    else:
        baseline_artist, conditional_artist = plot_per_target_profile_on_axis(
            axis,
            panel=panel,
            k=k,
            effect_source=effect_source,
            magnitude_metric=magnitude_metric,
            add_arrows=True,
            show_effect_labels=show_effect_annotations,
        )
        if baseline_artist is not None and conditional_artist is not None:
            source_label = format_annotation_display_label(panel.source_id)
            axis.legend(
                handles=[baseline_artist, conditional_artist],
                labels=["Global baseline", f"Following {source_label}"],
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, format="pdf")
    plt.close(figure)
    print(f"Wrote per-target profile for K={k} to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for per-target sequential-dynamics plots."""

    args = parse_args(argv)
    k: int = int(args.window_k[0])
    cond_id: Optional[str] = getattr(args, "cond_id", None)

    triple_targets: Optional[Set[str]] = None
    try:
        if cond_id:
            panel, triple_targets = _build_triple_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id=args.source_id,
                cond_id=cond_id,
                pairwise_targets_raw=args.target_id,
                conditional_targets_raw=getattr(
                    args,
                    "conditional_target_id",
                    None,
                ),
                order_by_effect_size=bool(args.order_by_effect_size),
            )
        else:
            panel = build_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id_raw=args.source_id,
                target_ids_raw=args.target_id,
                effect_source=args.effect_source,
                order_by_effect_size=bool(args.order_by_effect_size),
            )
    except ValueError as exc:
        print(str(exc))
        return 2

    # Load window-count statistics for summary table.
    matrix_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}_K{k}_matrix.csv",
    )
    pairwise_counts = _load_pairwise_window_counts(
        matrix_path,
        k,
        args.source_id,
        panel.targets,
    )
    triple_counts: Dict[str, Tuple[int, int]] = {}
    triple_beta: Dict[str, Tuple[float, float, float]] = {}
    if cond_id and triple_targets:
        triples_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}_K{k}_triples_cowindow.csv",
        )
        triple_counts = _load_triple_window_counts(
            triples_path,
            k,
            args.source_id,
            cond_id,
            triple_targets,
        )
        if args.effect_source == "beta":
            triple_beta = _load_triple_beta_stats(
                triples_path,
                k,
                args.source_id,
                cond_id,
                triple_targets,
            )

    # Print a compact summary table including event and window counts and
    # distinguishing pairwise versus triple-conditioned targets.
    print("Per-target K-window summary:")
    header = (
        f"{'target':40s} {'type':>8s} "
        f"{'baseline':>10s} {'conditional':>12s} "
        f"{'delta':>10s} {'RR':>10s} {'OR':>10s} "
        f"{'N_events':>10s} {'N_windows':>10s}"
    )
    print(header)

    if cond_id and triple_targets:
        # In conditional mode, report both pairwise and triple effects for
        # conditional targets. For pair rows, the effect compares P(Z|X)
        # against the global baseline P(Z). For triple rows, the effect
        # compares P(Z|X,Y) against P(Z|X).
        for target in panel.targets:
            base_p = float(panel.baseline_means[target])
            pair_trials, pair_successes = pairwise_counts.get(target, (0, 0))
            p_x = float(pair_successes) / float(pair_trials) if pair_trials > 0 else 0.0

            # Pairwise row: P(Z|X) vs P(Z).
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
                f"{delta_pair:10.3f} {rr_pair:10.3f} {or_pair:10.3f} "
                f"{pair_successes:10d} {pair_trials:10d}"
            )

            # Triple row: P(Z|X,Y) vs P(Z|X) when available.
            if target not in triple_targets:
                continue
            triple_trials, triple_successes = triple_counts.get(target, (0, 0))
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
                or_triple = (
                    cond_odds / base_odds
                    if base_odds > 0.0
                    else float(
                        "nan",
                    )
                )
            else:
                or_triple = float("nan")
            print(
                f"{target:40s} {'triple':>8s} "
                f"{p_x:10.3f} {p_xy:12.3f} "
                f"{delta_triple:10.3f} {rr_triple:10.3f} {or_triple:10.3f} "
                f"{triple_successes:10d} {triple_trials:10d}"
            )
    else:
        # Non-conditional mode: fall back to the standard single-step
        # baseline vs conditional summaries.
        for (
            target,
            base,
            cond,
            delta,
            risk_ratio,
            odds_ratio,
        ) in compute_effect_metrics(panel):
            trials, successes = pairwise_counts.get(target, (0, 0))
            print(
                f"{target:40s} {'pair':>8s} "
                f"{base:10.3f} {cond:12.3f} "
                f"{delta:10.3f} {risk_ratio:10.3f} {odds_ratio:10.3f} "
                f"{successes:10d} {trials:10d}"
            )

    _plot_per_target_bars(
        args.figure_path,
        k,
        panel,
        args.effect_source,
        args.magnitude_metric,
        show_effect_annotations=not bool(
            getattr(args, "hide_effect_annotations", False)
        ),
        cond_id=cond_id,
        triple_targets=triple_targets,
        pairwise_counts=pairwise_counts,
        triple_counts=triple_counts,
        triple_beta=triple_beta if triple_beta else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
