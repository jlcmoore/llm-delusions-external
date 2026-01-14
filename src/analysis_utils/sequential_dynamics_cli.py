"""Shared CLI utilities for sequential dynamics scripts.

This module contains small helpers that are reused by
``analysis/compute_sequential_annotation_dynamics.py`` and
``analysis/plot_sequential_annotation_dynamics.py`` to keep their argument
parsing logic simple and consistent.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, TextIO


def add_window_k_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--window-k`` argument to ``parser``.

    The flag may be provided multiple times to request several window sizes.
    When omitted, downstream helpers default to ``K = 0, 1, 10``.
    """

    parser.add_argument(
        "--window-k",
        type=int,
        action="append",
        help=(
            "Window size K in messages used when computing sequential "
            "dynamics. May be provided multiple times; when omitted, "
            "defaults to K = 0, 1, and 10."
        ),
    )


def parse_window_k_arguments(
    parser: argparse.ArgumentParser,
    raw_values: Optional[Sequence[int]],
    default_values: Sequence[int] = (0, 1, 10),
) -> List[int]:
    """Return a validated, sorted list of unique window sizes K.

    Parameters
    ----------
    parser:
        Argument parser used to report validation errors.
    raw_values:
        Raw ``--window-k`` values collected by argparse, or ``None`` when
        the flag was not provided by the caller.
    default_values:
        Default window sizes to use when ``raw_values`` is ``None``.

    Returns
    -------
    List[int]
        Sorted list of unique window sizes K.
    """

    if raw_values is None:
        raw_values = list(default_values)

    ks: List[int] = []
    for value in raw_values:
        if value < 0:
            parser.error("--window-k values must be non-negative")
        ks.append(int(value))

    if not ks:
        parser.error("At least one --window-k value is required")

    return sorted(set(ks))


def read_matrix_header(handle: TextIO) -> tuple[list[str], Dict[str, int]]:
    """Return header fields and name-to-index mapping for a matrix CSV.

    This helper reads a single header line from ``handle`` and constructs
    a mapping from column names to their integer indices, which is reused by
    sequential dynamics analysis and plotting scripts.
    """

    header = handle.readline().rstrip("\n").split(",")
    indices: Dict[str, int] = {name: index for index, name in enumerate(header)}
    return header, indices
