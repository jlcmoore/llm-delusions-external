"""Helpers for reading and matching annotation JSONL output files."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from utils.io import iter_jsonl_dicts

ParticipantMessageKey = Tuple[str, str, int, int]
SeenKey = tuple[str, str, int, int, str]
ReplayKey = ParticipantMessageKey
IndexPair = Tuple[int, int]


class ParticipantOrderingType(str, Enum):
    """Enumerated ordering categories for participant message timelines.

    Values
    ------
    FULL_DATED:
        Timestamps or conversation dates are sufficiently complete to support
        a global chronological ordering of all messages across conversations.
    GLOBAL_ORDER:
        Messages lack reliable timestamps but do expose a complete global
        ordering via indices such as ``chat_index`` and ``message_index``.
    CONVERSATION_ONLY:
        Ordering is only guaranteed within conversations, not across all
        conversations for a participant.
    UNKNOWN:
        The available metadata is insufficient to establish a consistent
        ordering even within conversations.
    """

    FULL_DATED = "full_dated"
    GLOBAL_ORDER = "global_order"
    CONVERSATION_ONLY = "conversation_only"
    UNKNOWN = "unknown"


@dataclass
class ParticipantOrderingInfo:
    """Summary of ordering metadata available for a participant.

    Parameters
    ----------
    participant:
        Canonical participant identifier.
    ordering_type:
        High-level ordering category for downstream analyses.
    has_timestamps:
        True when a substantial fraction of messages include a usable
        ``timestamp`` field.
    has_conversation_dates:
        True when a substantial fraction of conversations include a usable
        ``chat_date`` or equivalent conversation-level timestamp.
    has_message_indices:
        True when message-level indices (``chat_index`` and
        ``message_index``) are available for at least some records.
    """

    participant: str
    ordering_type: ParticipantOrderingType
    has_timestamps: bool
    has_conversation_dates: bool
    has_message_indices: bool


def iter_jsonl_meta(output_root: Path) -> Iterator[tuple[Path, dict]]:
    """Yield ``(path, meta_dict)`` for JSONL files that start with a meta line."""

    for dirpath, _dirnames, filenames in os.walk(output_root):
        for name in filenames:
            if not name.lower().endswith(".jsonl"):
                continue
            candidate_path = Path(dirpath) / name
            try:
                with candidate_path.open(
                    "r", encoding="utf-8", errors="ignore"
                ) as handle:
                    first = handle.readline()
            except OSError:
                continue
            try:
                meta = json.loads(first)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if isinstance(meta, dict) and meta.get("type") == "meta":
                yield candidate_path, meta


def iter_annotation_records(
    family_files: Sequence[Path],
    annotation_id: str,
) -> Iterator[Mapping[str, object]]:
    """Yield non-meta records for a single annotation across ``family_files``.

    Parameters
    ----------
    family_files:
        JSONL files belonging to a single annotation job family.
    annotation_id:
        Target annotation identifier to match in each record.

    Yields
    ------
    Mapping[str, object]
        Parsed JSON objects whose ``annotation_id`` field matches the target
        and whose ``type`` is not ``\"meta\"``.
    """

    for jsonl_path in sorted(family_files):
        for record in iter_jsonl_dicts(jsonl_path):
            if record.get("type") == "meta":
                continue
            annotation_raw = record.get("annotation_id")
            if not isinstance(annotation_raw, str):
                continue
            if annotation_raw != annotation_id:
                continue
            yield record


def iter_jsonl_dicts_ignoring_errors(jsonl_path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, ignoring decode errors.

    Parameters
    ----------
    jsonl_path:
        Path to a JSON Lines file whose rows may contain JSON objects.

    Yields
    ------
    dict
        Parsed JSON objects for non-empty lines that are dictionaries and do
        not trigger JSON parsing errors. Lines that cannot be parsed are
        skipped. File-level I/O errors are not suppressed.
    """

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            yield obj


def iter_jsonl_records(jsonl_path: Path) -> Iterator[dict]:
    """Yield non-meta JSON objects from a JSONL file.

    This helper builds on :func:`iter_jsonl_dicts_ignoring_errors` and skips
    any objects whose ``type`` field is equal to ``\"meta\"``.

    Parameters
    ----------
    jsonl_path:
        Path to a JSON Lines file whose rows may contain JSON objects.

    Yields
    ------
    dict
        Parsed JSON objects for non-empty lines that are dictionaries and do
        not have ``type == \"meta\"``.
    """

    for obj in iter_jsonl_dicts_ignoring_errors(jsonl_path):
        if obj.get("type") == "meta":
            continue
        yield obj


def get_annotation_id(obj: Mapping[str, object]) -> Optional[str]:
    """Return a cleaned annotation id string or ``None``.

    Parameters
    ----------
    obj:
        JSON-like mapping that may contain an ``annotation_id`` field.

    Returns
    -------
    Optional[str]
        Stripped annotation identifier when present and non-empty;
        otherwise ``None``.
    """

    annotation_raw = obj.get("annotation_id")
    if not isinstance(annotation_raw, str):
        return None
    annotation_id = annotation_raw.strip()
    return annotation_id or None


def parse_message_indices(obj: Mapping[str, object]) -> Optional[IndexPair]:
    """Return ``(chat_index, message_index)`` when both are usable integers.

    Parameters
    ----------
    obj:
        JSON object that may contain ``chat_index`` and ``message_index``.

    Returns
    -------
    Optional[IndexPair]
        Tuple of parsed indices or ``None`` when either value is missing or
        cannot be interpreted as an integer.
    """

    try:
        chat_index = int(obj.get("chat_index"))
        message_index = int(obj.get("message_index"))
    except (ValueError, TypeError):
        return None
    return chat_index, message_index


def classify_participant_ordering(
    records: Iterable[Mapping[str, object]],
    *,
    timestamp_coverage_threshold: float = 0.9,
    conversation_date_coverage_threshold: float = 0.9,
) -> Dict[str, ParticipantOrderingInfo]:
    """Classify participants by the strength of their message ordering metadata.

    This helper inspects a stream of annotation-like records and groups them by
    participant, using timestamp fields, conversation dates, and message indices
    to assign one of several ordering types. The resulting mapping can be used
    by both static analysis scripts and interactive dashboards to decide which
    forms of temporal visualization are appropriate for each participant.

    Parameters
    ----------
    records:
        Iterable of JSON-like mappings containing at least the fields used by
        :func:`build_participant_message_key` and :func:`extract_conversation_key`.
    timestamp_coverage_threshold:
        Minimum fraction (between 0 and 1) of messages that must include a
        usable ``timestamp`` field in order to treat a participant as
        ``FULL_DATED`` based on message-level timestamps.
    conversation_date_coverage_threshold:
        Minimum fraction (between 0 and 1) of conversations that must include
        a usable ``chat_date`` field in order to treat a participant as
        ``FULL_DATED`` based on conversation-level timestamps.

    Returns
    -------
    Dict[str, ParticipantOrderingInfo]
        Mapping from participant identifier to a summary of available ordering
        metadata and an assigned :class:`ParticipantOrderingType`.
    """

    def build_participant_ordering_info(
        participant: str,
        *,
        total_messages: int,
        total_conversations: int,
        has_indices: bool,
        timestamp_coverage: float,
        conversation_date_coverage: float,
    ) -> ParticipantOrderingInfo:
        """Return ordering metadata for a single participant."""

        has_any_activity = not (total_messages == 0 and total_conversations == 0)
        has_any_dates = (
            timestamp_coverage >= timestamp_coverage_threshold
            and timestamp_coverage > 0.0
        ) or (
            conversation_date_coverage >= conversation_date_coverage_threshold
            and conversation_date_coverage > 0.0
        )

        ordering_type = resolve_ordering_or_unknown(
            has_any_activity=has_any_activity,
            has_any_dates=has_any_dates,
            total_messages=total_messages,
            total_conversations=total_conversations,
            has_indices=has_indices,
        )

        return ParticipantOrderingInfo(
            participant=participant,
            ordering_type=ordering_type,
            has_timestamps=timestamp_coverage > 0.0,
            has_conversation_dates=conversation_date_coverage > 0.0,
            has_message_indices=has_indices,
        )

    per_participant_messages: Dict[str, Set[ParticipantMessageKey]] = {}
    per_participant_messages_with_timestamp: Dict[str, Set[ParticipantMessageKey]] = {}
    per_participant_conversations: Dict[str, Set[Tuple[str, int]]] = {}
    per_participant_conversations_with_date: Dict[str, Set[Tuple[str, int]]] = {}
    per_participant_has_indices: Dict[str, bool] = {}

    for record in records:
        message_key = build_participant_message_key(record)
        if message_key is not None:
            participant, _, _, _ = message_key
            per_participant_messages.setdefault(participant, set()).add(message_key)
            per_participant_has_indices[participant] = True

            raw_timestamp = record.get("timestamp")
            if isinstance(raw_timestamp, str) and raw_timestamp.strip():
                per_participant_messages_with_timestamp.setdefault(
                    participant, set()
                ).add(message_key)

        conversation_key = extract_conversation_key(record)
        if conversation_key is not None:
            (
                participant,
                transcript_rel_path,
                conversation_index,
                _conversation_label,
                conversation_date,
            ) = conversation_key
            conv_key = (transcript_rel_path, conversation_index)
            per_participant_conversations.setdefault(participant, set()).add(conv_key)
            if isinstance(conversation_date, str) and conversation_date.strip():
                per_participant_conversations_with_date.setdefault(
                    participant, set()
                ).add(conv_key)

    results: Dict[str, ParticipantOrderingInfo] = {}

    participants: Set[str] = set(per_participant_messages) | set(
        per_participant_conversations
    )
    for participant in sorted(participants):
        message_keys = per_participant_messages.get(participant, set())
        total_messages = len(message_keys)
        message_keys_with_timestamp = per_participant_messages_with_timestamp.get(
            participant, set()
        )
        total_with_timestamp = len(message_keys_with_timestamp)

        conversation_keys = per_participant_conversations.get(participant, set())
        total_conversations = len(conversation_keys)
        conversation_keys_with_date = per_participant_conversations_with_date.get(
            participant, set()
        )
        total_with_date = len(conversation_keys_with_date)

        has_indices = per_participant_has_indices.get(participant, False)

        timestamp_coverage = (
            float(total_with_timestamp) / float(total_messages)
            if total_messages > 0
            else 0.0
        )
        conversation_date_coverage = (
            float(total_with_date) / float(total_conversations)
            if total_conversations > 0
            else 0.0
        )

        results[participant] = build_participant_ordering_info(
            participant=participant,
            total_messages=total_messages,
            total_conversations=total_conversations,
            has_indices=has_indices,
            timestamp_coverage=timestamp_coverage,
            conversation_date_coverage=conversation_date_coverage,
        )

    return results


def resolve_basic_ordering_type(
    *,
    total_messages: int,
    total_conversations: int,
    has_indices: bool,
) -> ParticipantOrderingType:
    """Return a basic ordering type from counts and index availability.

    Parameters
    ----------
    total_messages:
        Total number of messages observed for the participant.
    total_conversations:
        Total number of conversations observed for the participant.
    has_indices:
        Whether per-message indices are available and can provide a global
        ordering when only a single transcript file is present.

    Returns
    -------
    ParticipantOrderingType
        One of ``GLOBAL_ORDER``, ``CONVERSATION_ONLY`` or ``UNKNOWN``
        depending on the available metadata.
    """

    if total_messages <= 0:
        return ParticipantOrderingType.UNKNOWN
    if has_indices:
        return ParticipantOrderingType.GLOBAL_ORDER
    if total_conversations > 0:
        return ParticipantOrderingType.CONVERSATION_ONLY
    return ParticipantOrderingType.UNKNOWN


def resolve_dated_or_basic_ordering_type(
    *,
    has_any_dates: bool,
    total_messages: int,
    total_conversations: int,
    has_indices: bool,
) -> ParticipantOrderingType:
    """Return an ordering type that prefers dated metadata when available.

    Parameters
    ----------
    has_any_dates:
        Whether any usable timestamp or conversation-date information is
        available for the participant.
    total_messages:
        Total number of messages observed for the participant.
    total_conversations:
        Total number of conversations observed for the participant.
    has_indices:
        Whether per-message indices are available and can provide a global
        ordering when only a single transcript file is present.

    Returns
    -------
    ParticipantOrderingType
        ``FULL_DATED`` when ``has_any_dates`` is true; otherwise the result of
        :func:`resolve_basic_ordering_type`.
    """

    if has_any_dates:
        return ParticipantOrderingType.FULL_DATED
    return resolve_basic_ordering_type(
        total_messages=total_messages,
        total_conversations=total_conversations,
        has_indices=has_indices,
    )


def resolve_ordering_or_unknown(
    *,
    has_any_activity: bool,
    has_any_dates: bool,
    total_messages: int,
    total_conversations: int,
    has_indices: bool,
) -> ParticipantOrderingType:
    """Return an ordering type, falling back to UNKNOWN when inactive.

    Parameters
    ----------
    has_any_activity:
        Whether any messages and conversations should be considered for the
        participant. Callers may incorporate additional file-level criteria.
    has_any_dates:
        Whether any usable timestamp or conversation-date information is
        available for the participant.
    total_messages:
        Total number of messages observed for the participant.
    total_conversations:
        Total number of conversations observed for the participant.
    has_indices:
        Whether per-message indices are available and can provide a global
        ordering when only a single transcript file is present.

    Returns
    -------
    ParticipantOrderingType
        ``UNKNOWN`` when ``has_any_activity`` is false; otherwise the result
        of :func:`resolve_dated_or_basic_ordering_type`.
    """

    if not has_any_activity:
        return ParticipantOrderingType.UNKNOWN
    return resolve_dated_or_basic_ordering_type(
        has_any_dates=has_any_dates,
        total_messages=total_messages,
        total_conversations=total_conversations,
        has_indices=has_indices,
    )


def build_participant_message_key(
    obj: Mapping[str, object],
) -> Optional[ParticipantMessageKey]:
    """Return a participant message key or None when fields are invalid.

    The key has the shape ``(participant, source_path, chat_index, message_index)``.
    ``participant`` is resolved from ``participant`` or ``ppt_id`` fields.
    """

    participant_raw = obj.get("participant") or obj.get("ppt_id")
    participant = str(participant_raw or "").strip()
    source_path = str(obj.get("source_path") or "").strip()
    pair = parse_message_indices(obj)
    if not participant or not source_path or pair is None:
        return None
    chat_index, message_index = pair
    return participant, source_path, chat_index, message_index


def extract_conversation_key(
    obj: Mapping[str, object],
) -> Optional[tuple[str, str, int, Optional[str], object]]:
    """Return conversation-level key fields extracted from a record.

    The returned tuple has the shape
    ``(participant, transcript_rel_path, chat_index, chat_key, chat_date)``.
    ``participant`` is resolved from ``participant`` or ``ppt_id`` fields and
    ``transcript_rel_path`` is resolved from ``source_path`` or
    ``source_file``.

    Parameters
    ----------
    obj:
        Classification-like record containing participant, source, and chat
        index fields.

    Returns
    -------
    Optional[tuple[str, str, int, Optional[str], object]]
        Conversation key fields when all required components are present and
        well-typed; otherwise ``None``.
    """

    participant_raw = obj.get("participant") or obj.get("ppt_id")
    participant = str(participant_raw or "").strip()
    if not participant:
        return None

    source_path_raw = obj.get("source_path") or obj.get("source_file")
    transcript_rel_path = str(source_path_raw or "").strip()
    if not transcript_rel_path:
        return None

    try:
        conversation_index = int(obj.get("chat_index"))
    except (TypeError, ValueError):
        return None

    conversation_key_raw = obj.get("chat_key")
    conversation_key: Optional[str] = None
    if conversation_key_raw is not None:
        text = str(conversation_key_raw).strip()
        conversation_key = text or None

    conversation_date = obj.get("chat_date")

    return (
        participant,
        transcript_rel_path,
        conversation_index,
        conversation_key,
        conversation_date,
    )


def load_replay_message_keys(replay_path: Path) -> Set[ReplayKey]:
    """Return message keys for all records contained in ``replay_path``.

    A replay key is ``(participant, source_path, chat_index, message_index)``
    aggregated across all annotation ids. Meta lines are ignored.
    """

    keys: Set[ReplayKey] = set()
    try:
        for obj in iter_jsonl_records(replay_path):
            key = build_participant_message_key(obj)
            if key is None:
                continue
            keys.add(key)
    except OSError:
        return set()
    return keys


def collect_replay_files_for_job(
    replay_path: Path,
    *,
    output_root: Path,
    include_all_participants: bool,
    read_annotation_ids: Optional[object] = None,
) -> List[Path]:
    """Return JSONL files that belong to the same replay job as ``replay_path``.

    When ``include_all_participants`` is False, only ``replay_path`` is
    returned. When True, the function discovers additional JSONL files under
    ``output_root`` whose filenames and annotation sets match ``replay_path``
    and returns the combined list.

    Parameters
    ----------
    replay_path:
        Canonical JSONL file path to replay from.
    output_root:
        Output directory that may contain additional participant JSONL files
        from the same job.
    include_all_participants:
        When True, attempt to discover sibling participant files to replay.
    read_annotation_ids:
        Optional callable used to extract annotation identifiers from a JSONL
        file. When omitted, no additional files are collected.
    """

    replay_files: List[Path] = [replay_path]
    if not include_all_participants:
        return replay_files
    if read_annotation_ids is None:
        return replay_files

    base_name = replay_path.name
    target_annotation_ids = read_annotation_ids(replay_path)
    if (
        not target_annotation_ids
        or not output_root.exists()
        or not output_root.is_dir()
    ):
        return replay_files

    for candidate_path, _meta in iter_jsonl_meta(output_root):
        if candidate_path == replay_path:
            continue
        if candidate_path.name != base_name:
            continue
        candidate_ids = read_annotation_ids(candidate_path)
        if candidate_ids != target_annotation_ids:
            continue
        replay_files.append(candidate_path)
    return replay_files


def load_resume_keys(
    resume_path: Path, target_annotation: Optional[str]
) -> tuple[Set[SeenKey], Optional[str]]:
    """Return a set of already-seen message keys from a resume JSONL.

    A key is ``(participant, source_path, chat_index, message_index, annotation_id)``.
    Only records matching ``target_annotation`` are included when provided. The
    second return value is reserved for future participant inference and is
    currently always ``None``.
    """

    seen: Set[SeenKey] = set()
    try:
        for obj in iter_jsonl_records(resume_path):
            annotation_id = str(obj.get("annotation_id") or "").strip()
            if target_annotation and annotation_id != target_annotation:
                continue
            participant = str(obj.get("participant") or obj.get("ppt_id") or "").strip()
            source_path = str(obj.get("source_path") or "").strip()
            pair = parse_message_indices(obj)
            if pair is None:
                continue
            chat_index, message_index = pair
            if not (participant and source_path and annotation_id):
                continue
            seen.add(
                (participant, source_path, chat_index, message_index, annotation_id)
            )
    except OSError:
        return set(), None
    return seen, None


def iter_records_with_error_filter(
    jsonl_path: Path,
    *,
    allowed_error_prefixes: Optional[Sequence[str]] = None,
    drop_other_errors: bool = False,
) -> Iterator[dict]:
    """Yield non-meta JSON objects from ``jsonl_path`` applying error filters.

    This helper mirrors :func:`iter_jsonl_records` but allows callers to
    control how records with an ``error`` field are handled. It is intended
    for downstream analysis scripts that need consistent inclusion or
    exclusion of particular error types (for example, keeping quote-mismatch
    errors while dropping other failures).

    Parameters
    ----------
    jsonl_path:
        Path to a JSON Lines file whose rows may contain JSON objects.
    allowed_error_prefixes:
        Optional sequence of string prefixes. When provided, records whose
        ``error`` field is a string starting with any of these prefixes are
        always yielded, even when ``drop_other_errors`` is True.
    drop_other_errors:
        When True, records with a non-empty ``error`` field that does not
        match any prefix in ``allowed_error_prefixes`` are skipped. When
        False, such records are yielded unchanged.

    Yields
    ------
    dict
        Parsed JSON objects that are dictionaries, do not have
        ``type == \"meta\"``, and satisfy the configured error policy.
    """

    prefixes: Sequence[str] = allowed_error_prefixes or []
    for obj in iter_jsonl_dicts_ignoring_errors(jsonl_path):
        if obj.get("type") == "meta":
            continue
        error_value = obj.get("error")
        if error_value:
            error_text = str(error_value)
            is_allowed = any(
                error_text.startswith(prefix) for prefix in prefixes if prefix
            )
            if not is_allowed and drop_other_errors:
                continue
        yield obj


def iter_family_records_with_error_filter(
    family_files: Sequence[Path],
    *,
    allowed_error_prefixes: Optional[Sequence[str]] = None,
    drop_other_errors: bool = False,
) -> Iterator[dict]:
    """Yield records from all ``family_files`` applying a shared error policy.

    Parameters
    ----------
    family_files:
        JSONL files that belong to a single classification job.
    allowed_error_prefixes:
        Optional sequence of prefixes for errors that should be retained even
        when ``drop_other_errors`` is True.
    drop_other_errors:
        When True, error records whose messages do not start with a permitted
        prefix are skipped.

    Yields
    ------
    dict
        Parsed JSON objects satisfying the configured error policy across all
        files in the family.
    """

    for jsonl_path in sorted(family_files):
        yield from iter_records_with_error_filter(
            jsonl_path,
            allowed_error_prefixes=allowed_error_prefixes,
            drop_other_errors=drop_other_errors,
        )


# Helpers for indexing LLM classification output runs under ``annotation_outputs/``.
#
# These utilities read only the leading meta record from each JSONL file so that
# other components (HTTP server endpoints, agreement scripts, or viewers) can
# reason about available runs without duplicating directory-scanning logic.


@dataclass(frozen=True)
class AnnotationOutputRun:
    """Metadata describing a single classification JSONL output.

    Parameters
    ----------
    path:
        Absolute path to the JSONL file.
    rel_path:
        Path to the JSONL file relative to the repository root.
    model:
        Model identifier recorded in the meta record.
    participants:
        List of participants mentioned in the meta record, if present.
    annotation_ids:
        Annotation identifiers included in the run.
    preceding_context:
        Number of preceding context messages used during classification, or
        ``None`` when not recorded.
    generated_at:
        Timestamp string from the meta record, if available.
    bucket:
        Top-level bucket under ``annotation_outputs/`` (for example,
        ``under_irb`` or ``human_line``) when derivable from the path.
    participant_dir:
        Participant directory derived from the path (for example, ``irb_07``)
        when present.
    arguments:
        Dictionary of non-default CLI arguments recorded in the meta record
        under the ``arguments`` key, when available. This typically mirrors
        the parameters used to launch the original classification run.
    """

    path: Path
    rel_path: Path
    model: str
    participants: List[str]
    annotation_ids: List[str]
    preceding_context: Optional[int]
    generated_at: Optional[str]
    bucket: Optional[str]
    participant_dir: Optional[str]
    arguments: Dict[str, object]


def _iter_meta_jsonl_files(root: Path) -> Iterable[tuple[Path, dict]]:
    """Yield JSONL paths and parsed meta records under ``root``.

    Only the first non-empty line of each ``*.jsonl`` file is inspected. When
    the line is not valid JSON or does not contain ``type == \"meta\"``, the
    file is skipped.
    """

    for path in root.rglob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                first_line = ""
                for line in handle:
                    first_line = line.strip()
                    if first_line:
                        break
            if not first_line:
                continue
            try:
                obj = json.loads(first_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("type") != "meta":
                continue
        except OSError:
            continue
        yield path, obj


def iter_annotation_output_runs(
    root: Path | str = "annotation_outputs",
) -> Iterable[AnnotationOutputRun]:
    """Yield :class:`AnnotationOutputRun` records discovered under a root.

    Parameters
    ----------
    root:
        Directory to scan for JSONL outputs. Defaults to ``annotation_outputs``.

    Returns
    -------
    Iterable[AnnotationOutputRun]
        One metadata record per JSONL file containing a valid meta line.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        return []

    repo_root = root_path.parent.resolve()

    runs: List[AnnotationOutputRun] = []
    for path, meta in _iter_meta_jsonl_files(root_path):
        try:
            rel_path = path.relative_to(repo_root)
        except ValueError:
            rel_path = path
        model_raw = meta.get("model") or ""
        model = str(model_raw).strip() if isinstance(model_raw, str) else ""
        ann_ids_raw = meta.get("annotation_ids") or []
        annotation_ids = [
            str(value).strip()
            for value in ann_ids_raw
            if isinstance(value, str) and value.strip()
        ]
        participants_raw = meta.get("participants") or []
        participants = [
            str(value).strip()
            for value in (participants_raw or [])
            if isinstance(value, str) and value.strip()
        ]
        preceding_raw = meta.get("preceding_context")
        try:
            preceding = int(preceding_raw) if preceding_raw is not None else None
        except (TypeError, ValueError):
            preceding = None
        generated_at_raw = meta.get("generated_at")
        generated_at = (
            str(generated_at_raw).strip()
            if isinstance(generated_at_raw, str) and generated_at_raw.strip()
            else None
        )

        arguments_raw = meta.get("arguments") or {}
        arguments: Dict[str, object] = {}
        if isinstance(arguments_raw, dict):
            for key, value in arguments_raw.items():
                key_str = str(key).strip()
                if not key_str:
                    continue
                arguments[key_str] = value

        # Derive bucket and participant directory when structured as
        # annotation_outputs/<bucket>/<participant>/file.jsonl.
        bucket: Optional[str] = None
        participant_dir: Optional[str] = None
        try:
            rel_from_root = rel_path.relative_to(root_path)
            parts = list(rel_from_root.parts)
            if len(parts) >= 1:
                bucket = parts[0]
            if len(parts) >= 2:
                participant_dir = parts[1]
        except ValueError:
            bucket = None
            participant_dir = None

        runs.append(
            AnnotationOutputRun(
                path=path,
                rel_path=rel_path,
                model=model,
                participants=participants,
                annotation_ids=annotation_ids,
                preceding_context=preceding,
                generated_at=generated_at,
                bucket=bucket,
                participant_dir=participant_dir,
                arguments=arguments,
            )
        )

    return runs


__all__ = [
    "iter_family_records_with_error_filter",
    "iter_records_with_error_filter",
    "AnnotationOutputRun",
    "iter_annotation_output_runs",
    "ParticipantMessageKey",
    "SeenKey",
    "ReplayKey",
    "collect_replay_files_for_job",
    "IndexPair",
    "iter_jsonl_meta",
    "iter_jsonl_records",
    "load_replay_message_keys",
    "load_resume_keys",
    "parse_message_indices",
    "extract_conversation_key",
]
