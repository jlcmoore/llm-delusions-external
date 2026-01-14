"""Shared plotting helpers for annotation-length style effects.

This module provides utilities for rendering sorted dot plots on a ratio
scale, mirroring the style used in ``analysis/plot_annotation_hazard_effects.py``.
The helpers are shared between analysis scripts that visualise how
annotations relate to conversation length or remaining messages.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils.labels import shorten_annotation_label
from analysis_utils.style import COLOR_BOUNDARY, annotation_color_for_label


def load_effect_rows(input_path: Path) -> List[Dict[str, str]]:
    """Return all rows from an effects CSV."""

    resolved = input_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Effects CSV not found at {resolved}")

    rows: List[Dict[str, str]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def save_figure(output_path: Path, fig: plt.Figure) -> None:
    """Expand, create parent directories, and save a Matplotlib figure."""

    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved)
    plt.close(fig)
    print(f"Wrote figure to {resolved}")


def extract_effect_triples(
    rows: Sequence[Mapping[str, str]],
    *,
    beta_key: str,
    se_key: str,
    annotation_key: str = "annotation_id",
) -> List[Tuple[str, float, float]]:
    """Return (annotation_id, beta, se) triples from a generic effects table.

    Rows with missing or non-finite beta/SE values are discarded. Callers can
    pre-filter ``rows`` to restrict to a particular model type or predictor
    before invoking this helper.
    """

    effects: List[Tuple[str, float, float]] = []
    for row in rows:
        annotation_id = str(row.get(annotation_key) or "").strip()
        if not annotation_id:
            continue

        beta_raw = row.get(beta_key)
        se_raw = row.get(se_key)
        try:
            beta = float(beta_raw) if beta_raw not in (None, "") else float("nan")
            se = float(se_raw) if se_raw not in (None, "") else float("nan")
        except (TypeError, ValueError):
            continue

        if not math.isfinite(beta) or not math.isfinite(se) or se <= 0.0:
            continue

        effects.append((annotation_id, beta, se))

    return effects


def split_effect_triples(
    triples: Sequence[Tuple[str, float, float]],
) -> Tuple[List[str], List[float], List[float]]:
    """Return separate (ids, betas, ses) lists from effect triples.

    This helper keeps the common unpacking pattern for (annotation_id, beta,
    se) triples in a single place.
    """

    if not triples:
        return [], [], []

    annotation_ids = [item[0] for item in triples]
    betas = [item[1] for item in triples]
    ses = [item[2] for item in triples]
    return annotation_ids, betas, ses


def select_symmetric_extreme_triples(
    triples: Sequence[Tuple[str, float, float]],
    *,
    max_bottom: int,
    max_top: int,
) -> List[Tuple[str, float, float]]:
    """Return triples for the most negative and positive effects.

    The returned list contains, in order, up to ``max_bottom`` triples with
    the most negative effects followed by up to ``max_top`` triples with the
    most positive effects. When both limits are non-positive the input
    ``triples`` are returned unchanged.
    """

    if not triples or (max_bottom <= 0 and max_top <= 0):
        return list(triples)

    negatives = [item for item in triples if item[1] < 0.0]
    positives = [item for item in triples if item[1] > 0.0]

    negatives.sort(key=lambda item: item[1])
    positives.sort(key=lambda item: item[1], reverse=True)

    bottom_subset: List[Tuple[str, float, float]] = []
    top_subset: List[Tuple[str, float, float]] = []
    if max_bottom > 0:
        bottom_subset = negatives[:max_bottom]
    if max_top > 0:
        top_subset = positives[:max_top]
        top_subset = list(reversed(top_subset))

    return bottom_subset + top_subset


def plot_ratio_dot_effects(
    output_path: Path,
    annotation_ids: Sequence[str],
    betas: Sequence[float],
    ses: Sequence[float],
    *,
    x_label: str,
    title: str,
) -> None:
    """Render and save a sorted dot plot on a ratio scale.

    Parameters
    ----------
    output_path:
        Destination path for the figure (PDF or other Matplotlib-supported
        format).
    annotation_ids:
        Annotation identifiers corresponding to each coefficient.
    betas:
        Coefficients on the log scale (for example, log-rate or
        log-remaining-length effects).
    ses:
        Standard errors associated with ``betas``.
    x_label:
        Label for the x-axis.
    title:
        Plot title.
    """

    if not annotation_ids or not betas or not ses:
        print("No usable effect estimates found for plotting.")
        return

    if len(annotation_ids) != len(betas) or len(annotation_ids) != len(ses):
        print("Mismatched annotation, beta, and SE lengths; skipping plot.")
        return

    betas_array = np.asarray(list(betas), dtype=float)
    ses_array = np.asarray(list(ses), dtype=float)

    # Transform to multiplicative ratios.
    ratios = np.exp(betas_array)
    lower = np.exp(betas_array - 1.96 * ses_array)
    upper = np.exp(betas_array + 1.96 * ses_array)

    with np.errstate(invalid="ignore"):
        yerr_lower = np.maximum(0.0, ratios - lower)
        yerr_upper = np.maximum(0.0, upper - ratios)
    yerr = np.vstack([yerr_lower, yerr_upper])

    labels = [shorten_annotation_label(aid) for aid in annotation_ids]
    colors = [annotation_color_for_label(aid) for aid in annotation_ids]
    indices = np.arange(len(annotation_ids), dtype=float)

    plt.switch_backend("Agg")
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fig_height = max(4.0, 0.25 * float(len(annotation_ids)))
    fig, ax = plt.subplots(figsize=(8.0, fig_height))

    # Plot annotations in the order supplied. Callers should sort betas and
    # annotation_ids so that the desired effects appear at the intended
    # positions along the vertical axis.
    for i, (y_pos, ratio, color) in enumerate(zip(indices, ratios, colors)):
        low = yerr[0, i]
        high = yerr[1, i]
        ax.errorbar(
            [ratio],
            [y_pos],
            xerr=[[low], [high]],
            fmt="o",
            color=color,
            ecolor=COLOR_BOUNDARY,
            elinewidth=0.8,
            capsize=2.0,
        )

    # Reference line at no-effect ratio 1.0.
    ax.axvline(1.0, color=COLOR_BOUNDARY, linestyle="--", linewidth=1.0)

    ax.set_yticks(indices)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    # Ensure the x-axis fully captures the confidence intervals as well as
    # the point estimates by basing limits on all finite bounds.
    all_values = np.concatenate([ratios, lower, upper])
    finite_mask = np.isfinite(all_values)
    if not np.any(finite_mask):
        xmin = 1.0
        xmax = 1.0
    else:
        finite_values = all_values[finite_mask]
        xmin = float(np.nanmin(finite_values))
        xmax = float(np.nanmax(finite_values))
    pad = max(0.05 * (xmax - xmin if xmax > xmin else 1.0), 0.05)
    ax.set_xlim(xmin - pad, xmax + pad)

    fig.tight_layout()
    fig.savefig(resolved)
    plt.close(fig)
    print(f"Wrote effects summary figure to {resolved}")
