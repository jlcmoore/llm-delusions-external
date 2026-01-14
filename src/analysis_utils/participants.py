"""Participant-level helpers for analysis scripts.

This module centralises shared participant configuration, including a
blocklist used to exclude specific participants from aggregate analyses.
"""

from __future__ import annotations

from typing import Set

EXCLUDED_PARTICIPANTS: Set[str] = {
    "hl_10",
    "hl_13",
    "irb_02",
    "irb_04",
    "irb_08",
    "irb_09",
    "irb_10",
    "irb_11",
    "irb_13",
    "irb_14",
}


def is_excluded_participant(participant: str) -> bool:
    """Return True when a participant id is on the exclusion list.

    Parameters
    ----------
    participant:
        Participant identifier string to check.

    Returns
    -------
    bool
        True when the normalised participant identifier is excluded.
    """

    if not participant:
        return False
    return participant.strip().lower() in EXCLUDED_PARTICIPANTS


def map_participant_for_display(participant: str) -> str:
    """Return a cohort-mapped participant label for outputs.

    Parameters
    ----------
    participant:
        Original participant identifier such as ``\"irb_05\"`` or ``\"hl_03\"``.

    Returns
    -------
    str
        Display label where ``irb_`` prefixes are mapped to a leading ``1-``
        and ``hl_`` prefixes are mapped to a leading ``2-``. Identifiers
        without these prefixes are returned unchanged.
    """

    if not participant:
        return participant
    cleaned = participant.strip()
    lower = cleaned.lower()
    if lower.startswith("irb_"):
        suffix = cleaned[4:]
        return f"1-{suffix}" if suffix else "1"
    if lower.startswith("hl_"):
        suffix = cleaned[3:]
        return f"2-{suffix}" if suffix else "2"
    return cleaned


__all__ = [
    "EXCLUDED_PARTICIPANTS",
    "is_excluded_participant",
    "map_participant_for_display",
]
