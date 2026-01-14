"""Shared visual styling constants and helpers for analysis plots and dashboards."""

from __future__ import annotations

import hashlib
from typing import Tuple

from matplotlib import colormaps

# Primary series colors.
COLOR_USER = "#2563eb"
COLOR_ASSISTANT = "#16a34a"

# Neutral guideline and annotation colors.
COLOR_BOUNDARY = "#9ca3af"
COLOR_TEXT_MUTED = "#6b7280"

# Emphasis colors for alerts or selected points.
COLOR_ERROR = "#ef4444"


def annotation_color_for_label(label: str) -> Tuple[float, float, float, float]:
    """Return a deterministic RGBA color for an annotation label.

    A fixed qualitative colormap is indexed using a stable hash of the label
    so that the same annotation id or category is rendered with the same
    color across figures and analyses.

    Parameters
    ----------
    label:
        Annotation identifier or category name used to seed the color choice.

    Returns
    -------
    Tuple[float, float, float, float]
        RGBA color tuple drawn from a qualitative Matplotlib colormap.
    """

    palette = colormaps["tab20"].colors
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(palette)
    return palette[index]


__all__ = [
    "COLOR_USER",
    "COLOR_ASSISTANT",
    "COLOR_BOUNDARY",
    "COLOR_TEXT_MUTED",
    "COLOR_ERROR",
    "annotation_color_for_label",
]
