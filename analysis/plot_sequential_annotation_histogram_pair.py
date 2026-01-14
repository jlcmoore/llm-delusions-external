"""Plot two per-target P(Y | X) profiles as histograms.

This module mirrors the structure of
``analysis/plot_sequential_annotation_bars_pair.py`` but focuses on a
single series per target: the conditional probability P(Y | X) for a
selected source annotation X and one or more target annotations Y.

For each source, the script renders a vertical bar-style histogram where
the height of each bar corresponds to P(Y | X) and bars are coloured
according to the source annotation. Global baseline probabilities,
odds/risk ratios, and per-target change annotations are intentionally
omitted so that the figure emphasises the conditional probabilities
alone.

The y-axis label and x-axis tick labels follow the same conventions as
the existing sequential bar plots so that figures remain comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils.labels import shorten_annotation_label
from analysis_utils.plot_effects_utils import save_figure
from analysis_utils.sequential_bars_utils import (
    PanelMetrics,
    build_panel_metrics,
    format_annotation_display_label,
    print_pairwise_effect_summary,
)
from analysis_utils.sequential_dynamics_cli import parse_window_k_arguments
from analysis_utils.style import annotation_color_for_label


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the paired P(Y | X) histogram plot.

    The argument structure closely follows
    ``plot_sequential_annotation_bars_pair.py`` so that command-line usage
    remains familiar. Only the pairwise X->Y matrix is consulted; options
    related to triple co-window statistics and magnitude metrics are
    omitted.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot two per-target conditional probability profiles P(Y | X) "
            "side by side as histograms from precomputed X->Y matrix CSV "
            "tables."
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
        / "sequential_enrichment_histogram_pair_K{K}.pdf",
        help=(
            "Destination PDF path for the paired per-target histogram. The "
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
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the paired histogram plotting script.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments to parse instead of
        ``sys.argv[1:]``. This is primarily intended for testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``window_k`` normalised to a single-element
        list.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    args.window_k = parse_window_k_arguments(parser, args.window_k)
    if len(args.window_k) != 1:
        parser.error("Exactly one --window-k value must be provided.")
    return args


def _build_panels(
    *,
    output_prefix: Path,
    k: int,
    effect_source: str,
    order_by_effect_size: bool,
    left_source_id: str,
    left_target_ids: Optional[Sequence[str]],
    right_source_id: str,
    right_target_ids: Optional[Sequence[str]],
) -> Tuple[PanelMetrics, PanelMetrics]:
    """Return PanelMetrics for the left and right sources.

    This helper mirrors the setup in the paired bar plot script but only
    relies on pairwise X->Y statistics. Any ValueError raised while
    loading metrics is propagated to the caller so that a clear message
    is printed and a non-zero exit code is returned.
    """

    left_panel = build_panel_metrics(
        output_prefix=output_prefix,
        k=k,
        source_id_raw=left_source_id,
        target_ids_raw=left_target_ids,
        effect_source=effect_source,
        order_by_effect_size=order_by_effect_size,
    )
    right_panel = build_panel_metrics(
        output_prefix=output_prefix,
        k=k,
        source_id_raw=right_source_id,
        target_ids_raw=right_target_ids,
        effect_source=effect_source,
        order_by_effect_size=order_by_effect_size,
    )
    return left_panel, right_panel


def _plot_histogram_panel(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    add_ylabel: bool,
) -> Optional[plt.Container]:
    """Render a per-target P(Y | X) histogram onto an axis.

    Bars are drawn for the conditional probabilities P(Y | X) only.
    Global baselines, odds ratios, and per-target difference annotations
    are omitted by design.

    Parameters
    ----------
    axis:
        Matplotlib axis onto which the histogram is rendered.
    panel:
        Per-source per-target metrics loaded from the X->Y matrix.
    k:
        Window size K in messages, used for the y-axis label when
        ``effect_source == "beta"``.
    effect_source:
        Effect-size source in use (``"beta"`` or ``"enrichment"``),
        controlling the y-axis label text.
    add_ylabel:
        When ``True``, set the shared y-axis label on this axis.

    Returns
    -------
    Optional[plt.Container]
        The bar container for the conditional series, or ``None`` when
        no targets are available.
    """

    if not panel.targets:
        return None

    x_positions = np.arange(len(panel.targets), dtype=float)

    conditional_y = []
    conditional_yerr = [[], []]
    for target in panel.targets:
        cond_mean = float(panel.conditional_means[target])
        cond_low, cond_high = panel.conditional_cis[target]
        conditional_y.append(cond_mean)
        conditional_yerr[0].append(cond_mean - cond_low)
        conditional_yerr[1].append(cond_high - cond_mean)

    conditional_y_array = np.asarray(conditional_y, dtype=float)
    conditional_yerr_array = np.asarray(conditional_yerr, dtype=float)

    if ":" in panel.source_id:
        base_source_id, _role = panel.source_id.split(":", 1)
    else:
        base_source_id = panel.source_id
    conditional_color = annotation_color_for_label(base_source_id)

    bar_container = axis.bar(
        x_positions,
        conditional_y_array,
        yerr=conditional_yerr_array,
        align="center",
        width=0.8,
        color=conditional_color,
        ecolor=conditional_color,
        alpha=0.9,
        linewidth=0.0,
        capsize=3.0,
        zorder=3,
    )

    short_labels = []
    for name in panel.targets:
        negated = False
        role = ""
        base_id = name

        if name.startswith("not:"):
            negated = True
            remainder = name[len("not:") :].strip()
            if ":" in remainder:
                base_id, role = remainder.split(":", 1)
            else:
                base_id = remainder
        elif ":" in name:
            base_id, role = name.split(":", 1)

        base_label = shorten_annotation_label(base_id)
        if role == "user":
            base_label = f"{base_label} (user)"
        elif role == "assistant":
            base_label = f"{base_label} (bot)"

        if negated:
            label = r"$\mathbf{not}$ " + base_label
        else:
            label = base_label

        short_labels.append(label.replace("-", "-\n"))

    axis.set_xticks(x_positions)
    axis.set_xticklabels(short_labels, rotation=0, ha="right", fontsize=8)

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

    return bar_container


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for paired P(Y | X) histogram plots.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments to parse instead of the
        default ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero on success or a non-zero error code when metrics could not
        be loaded or no targets were available for plotting.
    """

    plt.switch_backend("Agg")

    args = parse_args(argv)
    k: int = int(args.window_k[0])

    try:
        left_panel, right_panel = _build_panels(
            output_prefix=args.output_prefix,
            k=k,
            effect_source=args.effect_source,
            order_by_effect_size=bool(args.order_by_effect_size),
            left_source_id=args.left_source_id,
            left_target_ids=args.left_target_id,
            right_source_id=args.right_source_id,
            right_target_ids=args.right_target_id,
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    if not left_panel.targets or not right_panel.targets:
        print("At least one panel had no targets; nothing to plot.")
        return 1

    print_pairwise_effect_summary(left_panel, panel_label="left panel")
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
        sharey=True,
        gridspec_kw={"width_ratios": [left_width, right_width]},
    )

    left_artist = _plot_histogram_panel(
        axis_left,
        panel=left_panel,
        k=k,
        effect_source=args.effect_source,
        add_ylabel=True,
    )
    right_artist = _plot_histogram_panel(
        axis_right,
        panel=right_panel,
        k=k,
        effect_source=args.effect_source,
        add_ylabel=False,
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

    if left_artist is not None:
        label = f"Following {format_annotation_display_label(left_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(left_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if right_artist is not None:
        label = f"Following {format_annotation_display_label(right_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(right_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if legend_handles:
        label_count = len(legend_labels)
        if label_count <= 2:
            ncol = label_count
            legend_kwargs: Dict[str, object] = {"loc": "lower center"}
        else:
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
    else:
        label_count = 0

    figure.tight_layout()

    if label_count > 2:
        figure.subplots_adjust(
            left=0.08,
            right=0.99,
            bottom=0.45,
            top=0.89,
            wspace=0.15,
        )

    figure_path = args.figure_path
    if "{K}" in str(figure_path):
        figure_path = Path(str(figure_path).format(K=k))
    save_figure(figure_path, figure)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
