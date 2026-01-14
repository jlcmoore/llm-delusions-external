"""Plot per-annotation length effects from a summary table.

This script visualises the output of
``analysis/compute_annotation_length_effects.py`` by reading a summary CSV
and rendering a sorted dot plot with confidence intervals on a ratio scale.
The presentation mirrors the style of
``analysis/plot_annotation_hazard_effects.py`` and the GLMM plotting script.

It can show:

* A symmetric set of annotations with the most negative and most positive
  effects on length, controlled by ``--max-bottom`` and ``--max-top``; or
* A single block of annotations ordered by effect size using
  ``--max-annotations``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from analysis_utils.plot_effects_utils import (
    extract_effect_triples,
    load_effect_rows,
    plot_ratio_dot_effects,
    select_symmetric_extreme_triples,
    split_effect_triples,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the length-effects plotting script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot per-annotation length effects from a summary CSV produced by "
            "compute_annotation_length_effects.py."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/data/annotation_length_effects.csv"),
        help=(
            "Input CSV containing per-annotation length-effect estimates. "
            "This is typically the output of compute_annotation_length_effects.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figures/annotation_length_effects.pdf"),
        help=(
            "Output PDF path for the length-effects summary plot "
            "(default: analysis/figures/annotation_length_effects.pdf)."
        ),
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=37,
        help=(
            "Legacy option: maximum number of annotations to display in the "
            "plot, ordered by length slope from most negative to most "
            "positive (default: 37). Ignored when either --max-bottom or "
            "--max-top is positive."
        ),
    )
    parser.add_argument(
        "--max-bottom",
        type=int,
        default=0,
        help=(
            "Maximum number of annotations with the most negative length "
            "effects to display (default: 0, meaning none unless specified)."
        ),
    )
    parser.add_argument(
        "--max-top",
        type=int,
        default=0,
        help=(
            "Maximum number of annotations with the most positive length "
            "effects to display (default: 0, meaning none unless specified)."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the length-effects plotting script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted, arguments
        are read from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace populated with defaults.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.max_annotations <= 0:
        parser.error("--max-annotations must be a positive integer")
    if args.max_bottom < 0:
        parser.error("--max-bottom must be a non-negative integer")
    if args.max_top < 0:
        parser.error("--max-top must be a non-negative integer")

    return args


def _select_effects_for_plot(
    rows: Sequence[Mapping[str, str]],
    *,
    max_annotations: int,
    max_bottom: int,
    max_top: int,
) -> Tuple[List[str], List[float], List[float]]:
    """Return annotation ids, betas, and SEs for plotting.

    The function filters to rows with a usable length coefficient. When the
    underlying analysis used a log-length transform, ``beta_length`` and
    ``se_length`` correspond to differences on the log scale, so
    ``exp(beta_length)`` can be interpreted as a multiplicative effect on
    expected conversation length.
    """

    if not rows:
        return [], [], []

    effects = extract_effect_triples(
        rows,
        beta_key="beta_length",
        se_key="se_length",
        annotation_key="annotation_id",
    )

    if not effects:
        return [], [], []

    # If either max_bottom or max_top is specified, use the symmetric
    # extremes mode shared with other length-style plots; otherwise fall
    # back to legacy max_annotations behaviour.
    if max_bottom > 0 or max_top > 0:
        selected = select_symmetric_extreme_triples(
            effects,
            max_bottom=max_bottom,
            max_top=max_top,
        )
    else:
        # Order from largest to smallest effect so that annotations with the
        # strongest length-increasing slopes appear at the top of the plot.
        effects.sort(key=lambda pair: pair[1], reverse=True)
        selected = effects[:max_annotations]

    return split_effect_triples(selected)


def _plot_length_effects(
    output_path: Path,
    annotation_ids: Sequence[str],
    betas: Sequence[float],
    ses: Sequence[float],
) -> None:
    """Render and save a ratio-scale plot of per-annotation length effects."""

    x_label = "Ratio of expected conversation length (code present or not)"
    plot_ratio_dot_effects(
        output_path=output_path,
        annotation_ids=annotation_ids,
        betas=betas,
        ses=ses,
        x_label=x_label,
        title="",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for plotting annotation length effects."""

    args = parse_args(argv)
    try:
        rows = load_effect_rows(Path(args.input))
    except FileNotFoundError as err:
        print(str(err))
        return 1

    annotation_ids, betas, ses = _select_effects_for_plot(
        rows,
        max_annotations=int(args.max_annotations),
        max_bottom=int(args.max_bottom),
        max_top=int(args.max_top),
    )
    _plot_length_effects(Path(args.output), annotation_ids, betas, ses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
