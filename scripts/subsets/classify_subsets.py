"""Classify conversation subsets for dependency and cohesion criteria.

This script reads subset JSON files (as produced by ``make_subsets.py``),
submits each conversation window to an LLM classifier, and scores three
dimensions:

1. Reliance on prior conversations.
2. Reliance on large uploaded documents.
3. Overall cohesion (fraction of on-topic messages).

Each dimension is rated from 0 (clearly absent) to 10 (strongly present).

The JSON Lines output records these scores and metadata only. Any
``passes_quality_filters`` decision that combines scores with harmful
annotation presence is computed downstream by
``summarize_subset_harmfulness.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

import json_repair

from annotation.annotation_prompts import (
    THINK_CLOSE_TAG,
    THINK_OPEN_TAG,
    add_llm_common_arguments,
    build_cot_addendum,
    disable_litellm_logging,
    extract_first_choice_fields,
)
from llm_utils import print_cost_summary, safe_estimate_max_request_cost
from llm_utils.client import DEFAULT_CHAT_MODEL, LLMClientError, completion
from utils.cli import (
    add_model_argument,
    add_output_path_argument,
    add_participants_argument,
    add_subset_input_argument,
)
from utils.io import iter_jsonl_dicts
from utils.schema import (
    MESSAGE_CONTENT_KEY,
    MESSAGE_ROLE_KEY,
    META_FIELD_ARGUMENTS,
    META_FIELD_GENERATED_AT,
    META_FIELD_INPUT_DIR,
    META_FIELD_LABELS,
    META_FIELD_MODEL,
    META_FIELD_PARTICIPANTS,
    RECORD_FIELD_COMMENTS,
    RECORD_FIELD_CONVERSATION_ID,
    RECORD_FIELD_CONVERSATION_TITLE,
    RECORD_FIELD_LABEL,
    RECORD_FIELD_LLM_NOTES,
    RECORD_FIELD_MESSAGES_COUNT,
    RECORD_FIELD_MODEL,
    RECORD_FIELD_PARTICIPANT,
    RECORD_FIELD_ROW,
    RECORD_FIELD_SCORES,
    RECORD_FIELD_SOURCE_REL_PATH,
    RECORD_FIELD_SUBSET_REL_PATH,
    RECORD_FIELD_TYPE,
    RECORD_TYPE_META,
    RECORD_TYPE_SUBSET_QUALITY,
    SCORE_FIELD_COHESION,
    SCORE_FIELD_PRIOR,
    SCORE_FIELD_UPLOADED,
    SUBSET_INFO_COMMENTS,
    SUBSET_INFO_CONVERSATION_ID,
    SUBSET_INFO_CONVERSATION_TITLE,
    SUBSET_INFO_KEY,
    SUBSET_INFO_LABEL,
    SUBSET_INFO_PARTICIPANT,
    SUBSET_INFO_ROW,
    SUBSET_INFO_SOURCE_REL_PATH,
    SUBSET_MESSAGES_KEY,
)
from utils.utils import extract_non_default_arguments, normalize_arg_value

DEFAULT_INPUT_DIR = "subsets"
DEFAULT_OUTPUT = "subset_quality.jsonl"
DEFAULT_PLAN_CSV = "subsets.csv"
DEFAULT_SCORES_CSV = "subset_quality_scores.csv"
MAX_COMPLETION_TOKENS = 512

disable_litellm_logging()


@dataclass(frozen=True)
class SubsetLocation:
    """Location and identifying metadata for a subset JSON file.

    Parameters
    ----------
    path:
        Full filesystem path to the subset JSON file.
    rel_path:
        Path to the subset JSON file relative to the ``--input-dir`` root.
    """

    path: Path
    rel_path: str


@dataclass(frozen=True)
class SubsetRecord:
    """Loaded subset payload and derived location metadata.

    Parameters
    ----------
    location:
        Filesystem location information for the subset.
    info:
        ``subset_info`` dictionary from the JSON payload.
    messages:
        Flat list of message dictionaries forming the conversation window.
    """

    location: SubsetLocation
    info: Dict[str, object]
    messages: List[Dict[str, object]]


@dataclass(frozen=True)
class SubsetScores:
    """Scores returned from the LLM for a single subset."""

    prior_conversation_reliance: int
    uploaded_document_reliance: int
    cohesion: int
    notes: Optional[str] = None


class SubsetClassificationError(Exception):
    """Raised when a subset cannot be classified successfully."""


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for subset classification."""

    parser = argparse.ArgumentParser(
        description=(
            "Classify conversation subsets for reliance on prior chats, "
            "uploaded documents, and overall cohesion."
        )
    )

    add_subset_input_argument(
        parser,
        flag="--input-dir",
        default_input_dir=DEFAULT_INPUT_DIR,
    )
    add_output_path_argument(
        parser,
        default_path=DEFAULT_OUTPUT,
        help_text=("JSON Lines output file path " f"(default: {DEFAULT_OUTPUT})."),
    )
    parser.add_argument(
        "--existing-quality-json",
        type=Path,
        default=None,
        help=(
            "Optional existing subset_quality.jsonl file. When provided, "
            "subsets whose subset_rel_path already appears in that file are "
            "skipped so only newly-unscored subsets are classified."
        ),
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=Path(DEFAULT_PLAN_CSV),
        help=(
            "Original subsets plan CSV used to generate subsets "
            f"(default: {DEFAULT_PLAN_CSV})."
        ),
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path(DEFAULT_SCORES_CSV),
        help=(
            "Output CSV containing per-row subset scores aligned to the "
            f"plan CSV (default: {DEFAULT_SCORES_CSV})."
        ),
    )
    add_model_argument(parser, default_model=DEFAULT_CHAT_MODEL)
    add_participants_argument(
        parser,
        help_text=(
            "Restrict processing to subsets whose subset_info.participant matches any "
            "of these ids (repeatable). Defaults to all participants."
        ),
    )
    parser.add_argument(
        "--label",
        "-l",
        action="append",
        dest="labels",
        help=(
            "Restrict processing to subsets whose subset_info.label matches any of "
            "these values (repeatable). Defaults to all labels."
        ),
    )
    parser.add_argument(
        "--max-subsets",
        type=int,
        default=0,
        help="Optional cap on the number of subsets to classify (0 means no limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show the prepared prompt for the first matching subset and "
            "exit without calling the model or writing records."
        ),
    )
    add_llm_common_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and attach default mapping."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    defaults = {key: parser.get_default(key) for key in vars(args)}
    setattr(args, "_defaults", defaults)
    return args


def _iter_subset_files(root: Path) -> Iterator[SubsetLocation]:
    """Yield subset JSON files under ``root``."""

    resolved_root = root.expanduser().resolve()
    for file_path in sorted(resolved_root.rglob("*.json")):
        try:
            rel_path = str(file_path.relative_to(resolved_root))
        except ValueError:
            rel_path = str(file_path)
        yield SubsetLocation(path=file_path, rel_path=rel_path)


def _load_subset(location: SubsetLocation) -> Optional[SubsetRecord]:
    """Return a SubsetRecord for the given file when it matches the schema."""

    try:
        raw = location.path.read_text(encoding="utf-8")
    except OSError as err:
        logging.warning("Failed to read subset file %s: %s", location.path, err)
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as err:
        logging.warning("Invalid JSON in subset file %s: %s", location.path, err)
        return None

    if not isinstance(data, dict):
        return None

    info = data.get(SUBSET_INFO_KEY)
    messages = data.get(SUBSET_MESSAGES_KEY)
    if not isinstance(info, dict) or not isinstance(messages, list):
        return None

    return SubsetRecord(
        location=location,
        info=info,
        messages=[m for m in messages if isinstance(m, dict)],
    )


def _format_conversation(messages: Sequence[Mapping[str, object]]) -> str:
    """Return a human-readable representation of a conversation."""

    lines: List[str] = []
    total = len(messages)
    for index, message in enumerate(messages, start=1):
        raw_role = str(message.get(MESSAGE_ROLE_KEY) or "unknown")
        role = raw_role.strip().lower()
        if role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = raw_role or "Unknown"
        content = str(message.get(MESSAGE_CONTENT_KEY) or "")
        lines.append(f"{index:02d}. {label}: {content}")
        if index != total:
            lines.append("")
    return "\n".join(lines)


def _build_system_prompt() -> str:
    """Return the system prompt instructing the LLM how to score subsets."""

    lines: List[str] = [
        "You are a careful rater of short conversations.",
        "Your task is to evaluate a single conversation subset along three dimensions.",
        "",
        "For each dimension, produce an integer score from 0 to 10:",
        "- 0 means the property is clearly absent.",
        "- 5 means borderline or moderate presence.",
        "- 10 means the property is strongly and unambiguously present.",
        "",
        "Dimensions:",
        "1. prior_conversation_reliance",
        "   - High when the conversation depends heavily on earlier chats,",
        "     such as by summarizing or continuing prior discussions that are",
        "     not visible in the current subset.",
        '   - Look for explicit references like "as we discussed before",',
        '     "in our earlier conversation", or reliance on shared history',
        "     that would be confusing without prior context.",
        "",
        "2. uploaded_document_reliance",
        "   - High when the conversation focuses on a large uploaded document",
        "     (for example, a long PDF, article, book, or dataset).",
        "   - Look for summarization, close paraphrasing, or detailed analysis",
        "     of a multi-page document that the model or user refers to.",
        "",
        "3. cohesion",
        "   - High when the messages follow a single main topic with few",
        "     detours or unrelated tangents.",
        "   - Low when many messages are off-topic relative to the main",
        "     conversational stream.",
        "   - Treat conversations where more than about 25% of messages are",
        "     off-topic as low cohesion (score around 5 or below).",
        "",
        "Output exactly one JSON object with these fields:",
        '- "prior_conversation_reliance": integer 0–10,',
        '- "uploaded_document_reliance": integer 0–10,',
        '- "cohesion": integer 0–10,',
        'and an optional string field "notes" for brief comments.',
        "Return strictly valid JSON only, with no commentary, explanations,",
        "or code fences. Do not wrap the JSON in backticks or Markdown "
        "code fences; output just the JSON object.",
    ]
    return "\n".join(lines)


def _build_user_prompt(record: SubsetRecord, include_cot_addendum: bool) -> str:
    """Return the user prompt body for a single subset."""

    conversation_text = _format_conversation(record.messages)

    cot_block = build_cot_addendum() if include_cot_addendum else ""
    lines: List[str] = [
        "You are given a short conversation subset extracted from a larger transcript.",
        "",
        "Conversation (messages in order):",
        "```",
        conversation_text,
        "```",
        "",
        "Carefully read the conversation and assign integer scores from 0 to 10",
        "for the three dimensions described in the system prompt.",
        "",
        "Output format:",
        "- Return exactly one JSON object with the required integer fields.",
        '- Optionally include a short "notes" field if helpful.',
        "",
        cot_block,
        (
            "Remember: return only JSON, with no extra commentary, and do not "
            "wrap the JSON in backticks or Markdown code fences."
        ),
    ]
    return "\n".join(lines)


def _extract_scores_from_response(
    response: object,
) -> Tuple[Optional[str], SubsetScores]:
    """Parse a LiteLLM completion response into an optional scratchpad and scores."""

    try:
        content_raw, finish_reason = extract_first_choice_fields(response)
    except ValueError as err:
        raise SubsetClassificationError(f"{err} Response: {response}.") from err

    content = str(content_raw or "").strip()
    if not content:
        if finish_reason == "stop":
            raise SubsetClassificationError(
                "Empty content with finish_reason='stop'; expected JSON object."
            )
        raise SubsetClassificationError(
            f"Empty response from the LiteLLM API. Raw: {content_raw}"
        )

    text = content
    start = text.find(THINK_OPEN_TAG)
    end = -1
    if start != -1:
        end = text.find(THINK_CLOSE_TAG, start + len(THINK_OPEN_TAG))

    thought: Optional[str] = None
    response_for_parsing = text

    if start != -1 and end != -1:
        thought_segment = text[start + len(THINK_OPEN_TAG) : end]
        thought = thought_segment.strip() or None
        response_for_parsing = (
            text[:start] + text[end + len(THINK_CLOSE_TAG) :]
        ).strip()
    else:
        response_for_parsing = text.strip()

    try:
        parsed = json_repair.loads(response_for_parsing)
    except (json.JSONDecodeError, ValueError, TypeError) as err:
        raise SubsetClassificationError(
            f"Unable to parse model response as JSON: {err}"
        ) from err

    if not isinstance(parsed, dict):
        raise SubsetClassificationError(
            "Model response must be a JSON object with numeric fields "
            "'prior_conversation_reliance', 'uploaded_document_reliance', and "
            "'cohesion'."
        )

    def _coerce_score(key: str) -> int:
        raw = parsed.get(key)
        if not isinstance(raw, (int, float)):
            raise SubsetClassificationError(
                f"Field {key!r} must be a number between 0 and 10. "
                f"Response: {response_for_parsing}"
            )
        value = int(round(float(raw)))
        if value < 0 or value > 10:
            value = max(0, min(10, value))
        return value

    prior = _coerce_score("prior_conversation_reliance")
    uploaded = _coerce_score("uploaded_document_reliance")
    cohesion = _coerce_score("cohesion")

    raw_notes = parsed.get("notes")
    notes: Optional[str]
    if isinstance(raw_notes, str):
        notes = raw_notes.strip() or None
    else:
        notes = None

    return thought, SubsetScores(
        prior_conversation_reliance=prior,
        uploaded_document_reliance=uploaded,
        cohesion=cohesion,
        notes=notes,
    )


def _classify_single_subset(
    record: SubsetRecord,
    *,
    model: str,
    timeout: int,
    include_cot_addendum: bool,
) -> Tuple[SubsetScores, Optional[str]]:
    """Submit a single subset to the model and return scores and scratchpad."""

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(record, include_cot_addendum=include_cot_addendum)

    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt.rstrip()}\n\n"
                f"Do not use more than {MAX_COMPLETION_TOKENS} tokens in your response."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = completion(
            model=model,
            messages=messages,
            timeout=timeout,
            enable_reasoning_defaults=True,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    except LLMClientError as err:
        raise SubsetClassificationError(
            f"LiteLLM request failed for subset {record.location.rel_path}: {err}"
        ) from err

    thought, scores = _extract_scores_from_response(response)
    return scores, thought


def _iter_filtered_subsets(
    root: Path,
    *,
    participants: Optional[Sequence[str]],
    labels: Optional[Sequence[str]],
) -> Iterable[SubsetRecord]:
    """Yield loaded subsets filtered by participant and label."""

    participants_set = {p.strip() for p in participants} if participants else None
    labels_set = {l.strip() for l in labels} if labels else None

    for location in _iter_subset_files(root):
        record = _load_subset(location)
        if record is None:
            continue
        participant = str(record.info.get(SUBSET_INFO_PARTICIPANT) or "").strip()
        label = str(record.info.get(SUBSET_INFO_LABEL) or "").strip()
        if participants_set and participant not in participants_set:
            continue
        if labels_set and label not in labels_set:
            continue
        yield record


def _load_existing_scored_paths(path: Optional[Path]) -> set[str]:
    """Return subset_rel_path values that already have quality scores.

    When ``path`` is None or does not exist, an empty set is returned.
    """

    if path is None:
        return set()

    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return set()

    existing: set[str] = set()
    try:
        for obj in iter_jsonl_dicts(resolved):
            if obj.get(RECORD_FIELD_TYPE) != RECORD_TYPE_SUBSET_QUALITY:
                continue
            rel = str(obj.get(RECORD_FIELD_SUBSET_REL_PATH) or "").strip()
            if rel:
                existing.add(rel)
    except OSError:
        return set()
    return existing


def _open_output(path: Path) -> Tuple[Path, TextIO]:
    """Open the output JSONL file, writing a meta header when newly created."""

    resolved = path.expanduser().resolve()
    is_new = not resolved.exists()
    try:
        handle = resolved.open("a", encoding="utf-8")
    except OSError as err:
        raise OSError(f"Failed to open output file {resolved}: {err}") from err

    if is_new:
        meta = {
            RECORD_FIELD_TYPE: RECORD_TYPE_META,
            META_FIELD_GENERATED_AT: datetime.now().isoformat(),
            META_FIELD_MODEL: None,
            META_FIELD_PARTICIPANTS: None,
            META_FIELD_LABELS: None,
            META_FIELD_INPUT_DIR: None,
            META_FIELD_ARGUMENTS: None,
        }
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        handle.flush()

    return resolved, handle


def _build_subset_messages(
    record: SubsetRecord, include_cot_addendum: bool
) -> List[dict[str, str]]:
    """Return LiteLLM-style messages for a single subset."""

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(record, include_cot_addendum=include_cot_addendum)
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt.rstrip()}\n\n"
                f"Do not use more than {MAX_COMPLETION_TOKENS} tokens in your response."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _iter_dry_run_requests(
    first: SubsetRecord,
    remaining: Iterable[SubsetRecord],
    include_cot_addendum: bool,
) -> Iterator[tuple[str, str, List[dict[str, str]]]]:
    """Yield request payloads for cost estimation in dry-run mode."""

    yield (
        "subset-quality",
        "Subset quality filters",
        _build_subset_messages(first, include_cot_addendum),
    )
    for record in remaining:
        yield (
            "subset-quality",
            "Subset quality filters",
            _build_subset_messages(record, include_cot_addendum),
        )


def _run_dry_run(
    subset_iter: Iterator[SubsetRecord],
    *,
    model: str,
    include_cot_addendum: bool,
) -> int:
    """Print prompt preview and cost estimate for the first matching subset."""

    first = next(subset_iter, None)
    if first is None:
        print("No subsets matched the provided filters.", file=sys.stderr)
        return 1

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(first, include_cot_addendum=include_cot_addendum)
    separator = "-" * 80
    print(separator)
    print("System prompt:")
    print(separator)
    print(system_prompt)
    print()
    print(separator)
    print("User prompt for first matching subset:")
    print(separator)
    print(user_prompt)

    total_cost, cost_breakdown, max_tokens, total_requests = (
        safe_estimate_max_request_cost(
            model,
            _iter_dry_run_requests(first, subset_iter, include_cot_addendum),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            progress_callback=None,
        )
    )

    print_cost_summary(
        model,
        MAX_COMPLETION_TOKENS,
        total_cost,
        cost_breakdown,
        max_tokens,
        total_requests,
    )
    return 0


def _update_score_counters(
    prior_counts: Counter[int],
    uploaded_counts: Counter[int],
    cohesion_counts: Counter[int],
    scores: SubsetScores,
) -> None:
    """Increment per-dimension score counters for a single subset."""

    prior_counts[scores.prior_conversation_reliance] += 1
    uploaded_counts[scores.uploaded_document_reliance] += 1
    cohesion_counts[scores.cohesion] += 1


def _build_output_record(
    record: SubsetRecord,
    scores: SubsetScores,
    model: str,
) -> Dict[str, object]:
    """Return a JSON-serializable record for a classified subset."""

    row_value = record.info.get(SUBSET_INFO_ROW)
    try:
        row_number = int(row_value) if row_value is not None else None
    except (TypeError, ValueError):
        row_number = None

    output_record: Dict[str, object] = {
        RECORD_FIELD_TYPE: RECORD_TYPE_SUBSET_QUALITY,
        RECORD_FIELD_SUBSET_REL_PATH: record.location.rel_path,
        RECORD_FIELD_ROW: row_number,
        RECORD_FIELD_PARTICIPANT: record.info.get(SUBSET_INFO_PARTICIPANT),
        RECORD_FIELD_LABEL: record.info.get(SUBSET_INFO_LABEL),
        RECORD_FIELD_SOURCE_REL_PATH: record.info.get(SUBSET_INFO_SOURCE_REL_PATH),
        RECORD_FIELD_CONVERSATION_ID: record.info.get(SUBSET_INFO_CONVERSATION_ID),
        RECORD_FIELD_CONVERSATION_TITLE: record.info.get(
            SUBSET_INFO_CONVERSATION_TITLE
        ),
        RECORD_FIELD_MESSAGES_COUNT: len(record.messages),
        RECORD_FIELD_SCORES: {
            SCORE_FIELD_PRIOR: scores.prior_conversation_reliance,
            SCORE_FIELD_UPLOADED: scores.uploaded_document_reliance,
            SCORE_FIELD_COHESION: scores.cohesion,
        },
        RECORD_FIELD_MODEL: model,
    }
    comments = record.info.get(SUBSET_INFO_COMMENTS)
    if comments is not None:
        output_record[RECORD_FIELD_COMMENTS] = comments
    if scores.notes:
        output_record[RECORD_FIELD_LLM_NOTES] = scores.notes
    return output_record


def _log_score_distributions(
    prior_counts: Counter[int],
    uploaded_counts: Counter[int],
    cohesion_counts: Counter[int],
    processed: int,
) -> None:
    """Log per-dimension score histograms when at least one subset succeeded."""

    if processed <= 0:
        return

    logging.info(
        "Score distribution (prior_conversation_reliance): %s",
        dict(sorted(prior_counts.items())),
    )
    logging.info(
        "Score distribution (uploaded_document_reliance): %s",
        dict(sorted(uploaded_counts.items())),
    )
    logging.info(
        "Score distribution (cohesion): %s",
        dict(sorted(cohesion_counts.items())),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point for subset classification."""

    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_dir: Path = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("Input directory not found: %s", input_dir)
        return 2

    participants_filter: Optional[Sequence[str]] = args.participants
    labels_filter: Optional[Sequence[str]] = args.labels
    max_subsets: Optional[int] = args.max_subsets if args.max_subsets > 0 else None

    resolved_output, handle = _open_output(args.output)
    logging.info("Writing subset quality records to %s", resolved_output)

    # Update the meta line if this is an empty file we just created.
    try:
        handle.seek(0)
        first_line = handle.readline()
    except OSError:
        first_line = ""

    if first_line:
        try:
            meta = json.loads(first_line)
        except (json.JSONDecodeError, TypeError):
            meta = None
    else:
        meta = None

    if isinstance(meta, dict) and meta.get(RECORD_FIELD_TYPE) == RECORD_TYPE_META:
        meta[META_FIELD_MODEL] = args.model
        meta[META_FIELD_PARTICIPANTS] = participants_filter
        meta[META_FIELD_LABELS] = labels_filter
        meta[META_FIELD_INPUT_DIR] = str(input_dir)
        defaults = getattr(args, "_defaults", {})
        arguments = extract_non_default_arguments(args, defaults)
        meta[META_FIELD_ARGUMENTS] = {
            key: normalize_arg_value(value) for key, value in arguments.items()
        }
        try:
            handle.seek(0)
            handle.truncate(0)
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
            handle.flush()
        except OSError as err:
            logging.error(
                "Failed to update meta header in %s: %s", resolved_output, err
            )
            handle.close()
            return 2

    existing_paths = _load_existing_scored_paths(args.existing_quality_json)
    if existing_paths:
        logging.info(
            "Skipping %d already-scored subsets from %s",
            len(existing_paths),
            args.existing_quality_json,
        )

    base_iter = _iter_filtered_subsets(
        input_dir,
        participants=participants_filter,
        labels=labels_filter,
    )

    if existing_paths:
        subset_iter: Iterable[SubsetRecord] = (
            record
            for record in base_iter
            if record.location.rel_path not in existing_paths
        )
    else:
        subset_iter = base_iter

    if args.dry_run:
        exit_code = _run_dry_run(
            iter(subset_iter),
            model=args.model,
            include_cot_addendum=bool(args.cot),
        )
        handle.close()
        return exit_code

    # Main classification loop: score each subset and emit JSONL records.
    processed = 0
    errors = 0
    prior_counts: Counter[int] = Counter()
    uploaded_counts: Counter[int] = Counter()
    cohesion_counts: Counter[int] = Counter()
    try:
        for record in subset_iter:
            if max_subsets is not None and processed >= max_subsets:
                break

            try:
                scores, thought = _classify_single_subset(
                    record,
                    model=args.model,
                    timeout=args.timeout,
                    include_cot_addendum=bool(args.cot),
                )
            except SubsetClassificationError as err:
                logging.error("Subset %s failed: %s", record.location.rel_path, err)
                errors += 1
                continue

            _update_score_counters(
                prior_counts,
                uploaded_counts,
                cohesion_counts,
                scores,
            )

            output_record = _build_output_record(
                record,
                scores,
                args.model,
            )
            if thought:
                output_record["thought"] = thought

            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()

            processed += 1
            if processed % 10 == 0:
                logging.info("Processed %d subsets so far...", processed)

    finally:
        handle.close()

    _log_score_distributions(
        prior_counts,
        uploaded_counts,
        cohesion_counts,
        processed,
    )

    logging.info(
        "Completed subset classification: %d processed, %d errors.", processed, errors
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
