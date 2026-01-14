"""Helpers for loading per-annotation LLM score cutoffs.

This module centralizes logic for interpreting cutoff mappings so that both
scripts and library code can share the same semantics when applying
per-annotation thresholds to LLM scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from analysis_utils.labels import ROLE_SPLIT_BASE_IDS

LOGGER = logging.getLogger(__name__)


def load_llm_cutoffs_from_json(json_path: Optional[str]) -> Optional[Dict[str, int]]:
    """Return per-annotation LLM cutoffs loaded from a JSON file.

    Parameters
    ----------
    json_path:
        Path to a JSON file that either contains a simple mapping from
        annotation id to integer cutoff or a full metrics JSON payload with a
        top-level ``llm_score_cutoffs_by_annotation`` object.

    Returns
    -------
    Optional[Dict[str, int]]
        Mapping from annotation id to integer cutoff when the file could be
        read and parsed successfully; otherwise ``None``. Errors are logged
        using the module logger.
    """

    if not json_path:
        return None

    cutoffs_path = Path(str(json_path)).expanduser()
    try:
        text = cutoffs_path.read_text(encoding="utf-8")
    except OSError as err:
        LOGGER.error("Failed to read LLM cutoffs file %s: %s", cutoffs_path, err)
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as err:
        LOGGER.error(
            "Failed to parse LLM cutoffs file %s as JSON: %s",
            cutoffs_path,
            err,
        )
        return None
    if not isinstance(raw, dict):
        LOGGER.error(
            "LLM cutoffs file %s must contain a JSON object mapping "
            "annotation ids to integer cutoffs or a metrics payload with "
            "'llm_score_cutoffs_by_annotation'.",
            cutoffs_path,
        )
        return None

    # Allow either a plain mapping {annotation_id: cutoff} or a full metrics
    # JSON file that contains a top-level "llm_score_cutoffs_by_annotation"
    # object.
    if "llm_score_cutoffs_by_annotation" in raw and isinstance(
        raw.get("llm_score_cutoffs_by_annotation"), dict
    ):
        mapping_candidate = raw.get("llm_score_cutoffs_by_annotation") or {}
    else:
        mapping_candidate = raw

    if not isinstance(mapping_candidate, dict):
        LOGGER.error(
            "LLM cutoffs file %s must contain either a simple JSON object "
            "mapping annotation ids to integer cutoffs or a metrics JSON with "
            "a 'llm_score_cutoffs_by_annotation' object.",
            cutoffs_path,
        )
        return None

    cutoffs: Dict[str, int] = {}
    for key, value in mapping_candidate.items():
        if not isinstance(key, str):
            continue
        annotation_id = key.strip()
        if not annotation_id:
            continue
        try:
            cutoff_value = int(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Ignoring non-integer cutoff %r for annotation %r in LLM "
                "cutoffs file %s",
                value,
                annotation_id,
                cutoffs_path,
            )
            continue
        cutoffs[annotation_id] = cutoff_value

    if not cutoffs:
        LOGGER.error(
            "LLM cutoffs file %s did not contain any usable "
            "annotation->cutoff mappings.",
            cutoffs_path,
        )
        return None

    return cutoffs


def load_cutoffs_mapping(json_path: Optional[str]) -> Dict[str, int]:
    """Return a mapping from annotation id to cutoff or an empty mapping.

    This thin wrapper avoids repeated boilerplate in analysis scripts and keeps
    fallback behaviour centralised.
    """

    cutoffs = load_llm_cutoffs_from_json(json_path)
    if cutoffs is None:
        return {}

    mapping = dict(cutoffs)

    # Provide synthetic role-specific cutoffs for selected annotations so
    # that downstream analyses can safely consume derived score columns
    # (for example, ``score__user-platonic-affinity``) without requiring
    # separate entries in the metrics JSON. These synthetic ids inherit
    # the same cutoff as their base annotation.
    for base_id in ROLE_SPLIT_BASE_IDS:
        if base_id not in mapping:
            continue
        base_cutoff = mapping[base_id]
        for role_prefix in ("user", "assistant"):
            synthetic_id = f"{role_prefix}-{base_id}"
            mapping.setdefault(synthetic_id, base_cutoff)

    return mapping


__all__ = ["load_llm_cutoffs_from_json", "load_cutoffs_mapping"]
