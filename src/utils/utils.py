"""Common script utilities

These helpers cover filesystem, naming, JSONL iteration, and selection
conveniences shared by multiple scripts in this repo.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set

from annotation.classify_messages import ConversationKey
from chat.chat_utils import MessageContext


def slugify(text: str) -> str:
    """Return a filesystem-friendly representation of text."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return sanitized.strip("_") or "unnamed"


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not exist.

    Parameters:
        path: Directory path to create if missing.

    Returns:
        None.
    """

    os.makedirs(path, exist_ok=True)


def allocate_per_participant(
    sizes: Mapping[str, int],
    total_sample_size: int,
    *,
    equal: bool,
) -> Dict[str, int]:
    """Return per-participant allocations for a target sample size.

    Parameters
    ----------
    sizes:
        Mapping from participant identifier to the number of available items.
    total_sample_size:
        Target total number of items to allocate across all participants.
    equal:
        When True, aim to allocate the same number of items to each
        participant (as evenly as possible). When False, allocate in
        proportion to each participant's available item count.

    Returns
    -------
    Dict[str, int]
        Mapping from participant identifier to the number of items that
        should be sampled for that participant, never exceeding the
        corresponding entry in ``sizes``.
    """

    participants = sorted(sizes.keys())
    if not participants or total_sample_size <= 0:
        return {name: 0 for name in participants}

    total_messages = sum(max(0, sizes.get(name, 0)) for name in participants)
    if total_messages <= 0:
        return {name: 0 for name in participants}

    if total_sample_size >= total_messages:
        return {name: max(0, sizes.get(name, 0)) for name in participants}

    allocations: Dict[str, int] = {name: 0 for name in participants}

    if equal:
        base = total_sample_size // len(participants)
        for name in participants:
            allocations[name] = min(base, max(0, sizes.get(name, 0)))
        remainder = total_sample_size - sum(allocations.values())
        if remainder > 0:
            remaining_cap = sum(
                max(0, sizes.get(name, 0) - allocations[name]) for name in participants
            )
            remainders: list[tuple[str, float]] = []
            if remaining_cap > 0:
                for name in participants:
                    cap = max(0, sizes.get(name, 0) - allocations[name])
                    if remaining_cap > 0:
                        quota = remainder * (cap / float(remaining_cap))
                    else:
                        quota = 0.0
                    k = int(quota)
                    k = min(k, cap)
                    allocations[name] += k
                    remainder -= k
                    remainders.append((name, quota - k))
            if remainder > 0:
                remainders.sort(key=lambda item: item[1], reverse=True)
                for name, _frac in remainders:
                    if remainder == 0:
                        break
                    if allocations[name] < max(0, sizes.get(name, 0)):
                        allocations[name] += 1
                        remainder -= 1
        if remainder > 0:
            for name in participants:
                if remainder == 0:
                    break
                if allocations[name] < max(0, sizes.get(name, 0)):
                    allocations[name] += 1
                    remainder -= 1
    else:
        total = float(total_messages)
        remainders: list[tuple[str, float]] = []
        used = 0
        for name in participants:
            size_value = max(0, sizes.get(name, 0))
            quota = total_sample_size * (size_value / total)
            k = int(quota)
            k = min(k, size_value)
            allocations[name] = k
            used += k
            remainders.append((name, quota - k))
        remaining = total_sample_size - used
        if remaining > 0:
            remainders.sort(key=lambda item: item[1], reverse=True)
            for name, _frac in remainders:
                if remaining == 0:
                    break
                if allocations[name] < max(0, sizes.get(name, 0)):
                    allocations[name] += 1
                    remaining -= 1
        if remaining > 0:
            for name in participants:
                if remaining == 0:
                    break
                if allocations[name] < max(0, sizes.get(name, 0)):
                    allocations[name] += 1
                    remaining -= 1

    return allocations


def sample_messages_by_participant(
    message_iter: Iterator[MessageContext],
    sample_size: int,
    rng: random.Random,
    *,
    equal: bool = False,
) -> List[MessageContext]:
    """Sample messages with per-participant allocation.

    Parameters
    ----------
    message_iter:
        Source iterator yielding :class:`MessageContext` instances.
    sample_size:
        Maximum total number of messages to sample.
    rng:
        Random number generator used for participant order and index selection.
    equal:
        When True, sample the same number from each participant (as evenly as
        possible). When False, sample in proportion to each participant's total
        message count.

    Returns
    -------
    List[MessageContext]
        Selected contexts preserving the original global iteration order.
    """

    if sample_size <= 0:
        return []

    buckets, all_collected = _collect_messages_by_participant(message_iter)
    total_messages = len(all_collected)
    if total_messages == 0:
        return []
    if sample_size >= total_messages:
        return [ctx for _seq, ctx in all_collected]

    chosen_pairs = _allocate_sampled_pairs(
        buckets,
        sample_size,
        rng,
        equal=equal,
    )

    chosen_pairs.sort(key=lambda pair: pair[0])
    return [ctx for _seq, ctx in chosen_pairs]


def _collect_messages_by_participant(
    message_iter: Iterator[MessageContext],
) -> tuple[
    dict[str, List[tuple[int, MessageContext]]], List[tuple[int, MessageContext]]
]:
    """Return per-participant message buckets and global ordering.

    Messages are grouped by participant while also tracking their original
    global sequence index so that sampled results can later be restored to
    the input order.
    """

    buckets: dict[str, List[tuple[int, MessageContext]]] = {}
    all_collected: List[tuple[int, MessageContext]] = []
    for sequence_index, context in enumerate(message_iter):
        bucket_list = buckets.setdefault(context.participant, [])
        pair = (sequence_index, context)
        bucket_list.append(pair)
        all_collected.append(pair)
    return buckets, all_collected


def _allocate_sampled_pairs(
    buckets: Mapping[str, List[tuple[int, MessageContext]]],
    sample_size: int,
    rng: random.Random,
    *,
    equal: bool,
) -> List[tuple[int, MessageContext]]:
    """Return sampled (sequence index, context) pairs across participants.

    The allocation per participant is computed using
    :func:`allocate_per_participant` so that proportional and equal sampling
    semantics stay consistent across tools. Sampling within each participant
    is performed without replacement.
    """

    participants = sorted(buckets.keys())
    sizes = {participant: len(buckets[participant]) for participant in participants}
    allocations = allocate_per_participant(sizes, sample_size, equal=equal)

    chosen_pairs: List[tuple[int, MessageContext]] = []
    for participant in participants:
        allocation = allocations[participant]
        if allocation <= 0:
            continue
        participant_pairs = buckets[participant]
        if allocation >= len(participant_pairs):
            chosen_pairs.extend(participant_pairs)
            continue
        sampled_indices = rng.sample(range(len(participant_pairs)), allocation)
        for index in sampled_indices:
            chosen_pairs.append(participant_pairs[index])

    return chosen_pairs


def sample_conversations_within_participant(
    message_iter: Iterator[MessageContext],
    sample_size: int,
    rng: random.Random,
) -> List[MessageContext]:
    """Return contexts sampled by selecting random conversations per participant.

    Parameters
    ----------
    message_iter:
        Source iterator yielding message contexts.
    sample_size:
        Maximum number of messages to include across sampled conversations.
    rng:
        Random number generator used for sampling.

    Returns
    -------
    List[MessageContext]
        Sampled contexts with length at most ``sample_size`` preserving the
        original conversation order. When ``sample_size`` is zero or negative,
        all conversations are included in randomized order.
    """

    conversation_messages: dict[ConversationKey, List[MessageContext]] = {}
    conversation_order: dict[ConversationKey, int] = {}
    participant_to_conversations: defaultdict[str, List[ConversationKey]] = defaultdict(
        list
    )

    for index, context in enumerate(message_iter):
        conversation_key: ConversationKey = (
            context.participant,
            context.source_path,
            context.chat_key,
            context.chat_index,
        )
        if conversation_key not in conversation_messages:
            conversation_messages[conversation_key] = []
            conversation_order[conversation_key] = index
            participant_to_conversations[context.participant].append(conversation_key)
        conversation_messages[conversation_key].append(context)

    if not conversation_messages:
        return []

    include_all = sample_size <= 0
    participants = list(participant_to_conversations.keys())
    rng.shuffle(participants)

    selected_conversations: List[ConversationKey] = []
    remaining = sample_size
    for participant in participants:
        conversation_keys = participant_to_conversations[participant][:]
        rng.shuffle(conversation_keys)
        for conversation_key in conversation_keys:
            if include_all:
                selected_conversations.append(conversation_key)
                continue
            conversation_length = len(conversation_messages[conversation_key])
            if conversation_length > remaining and selected_conversations:
                continue
            selected_conversations.append(conversation_key)
            remaining = max(remaining - conversation_length, 0)
            if remaining == 0:
                break
        if not include_all and remaining == 0:
            break

    if not include_all and not selected_conversations:
        fallback_key = min(
            conversation_messages,
            key=lambda candidate: len(conversation_messages[candidate]),
        )
        selected_conversations.append(fallback_key)

    selected_conversations.sort(key=lambda key: conversation_order[key])

    sampled_contexts: List[MessageContext] = []
    for conversation_key in selected_conversations:
        for context in conversation_messages[conversation_key]:
            if not include_all and len(sampled_contexts) >= sample_size:
                return sampled_contexts
            sampled_contexts.append(context)

    return sampled_contexts


def limit_conversations_by_participant(
    message_iter: Iterator[MessageContext],
    max_conversations: int,
) -> Iterator[MessageContext]:
    """Yield contexts limited to the first ``max_conversations`` per participant."""

    if max_conversations <= 0:
        yield from message_iter
        return

    participant_counts: defaultdict[str, int] = defaultdict(int)
    allowed_keys: Set[ConversationKey] = set()
    skipped_keys: Set[ConversationKey] = set()

    for context in message_iter:
        conversation_key: ConversationKey = (
            context.participant,
            context.source_path,
            context.chat_key,
            context.chat_index,
        )
        if conversation_key in allowed_keys:
            yield context
            continue
        if conversation_key in skipped_keys:
            continue

        if participant_counts[context.participant] >= max_conversations:
            skipped_keys.add(conversation_key)
            continue

        participant_counts[context.participant] += 1
        allowed_keys.add(conversation_key)
        yield context


def pick_latest_per_parent(candidates: Sequence[Path]) -> List[Path]:
    """Select the most recently modified file per parent directory.

    Parameters
    ----------
    candidates: Sequence[Path]
        Collection of filesystem paths to consider. Multiple paths may share
        the same parent directory.

    Returns
    -------
    List[Path]
        One path per parent directory corresponding to the most recently
        modified candidate within that directory.
    """

    latest_by_parent: Dict[Path, Path] = {}
    for path in candidates:
        parent = path.parent
        current = latest_by_parent.get(parent)
        try:
            mtime = path.stat().st_mtime
            current_mtime = current.stat().st_mtime if current else -1.0
        except OSError:
            continue
        if current is None or mtime > current_mtime:
            latest_by_parent[parent] = path
    return list(latest_by_parent.values())


def short_slug(text: str, max_words: int = 4, max_len: int = 48) -> str:
    """Create a compact slug using up to ``max_words`` from text.

    The slug contains only lowercase alphanumeric characters and hyphens; no
    spaces. The result is truncated to ``max_len`` characters.

    Parameters:
        text: Input text to shorten and slugify.
        max_words: Maximum words to include.
        max_len: Maximum total length; truncated if necessary.

    Returns:
        A short, hyphen-separated slug suitable for filenames.
    """

    words = [w for w in text.strip().split() if w]
    chosen = words[: max(1, max_words)]
    cleaned = []
    for w in chosen:
        letters = [c.lower() for c in w if c.isalnum()]
        token = "".join(letters)
        if token:
            cleaned.append(token)
    if not cleaned:
        cleaned = ["untitled"]
    slug = "-".join(cleaned)
    return slug[:max_len].rstrip("-")


def pick_title_string(
    data: Dict[str, Any], conv_title: Optional[str], rel_path: str
) -> str:
    """Choose a best-effort title string for naming outputs.

    Parameters:
        data: Source JSON-like dictionary which may contain a ``meta`` block.
        conv_title: Title from selected conversation (if any).
        rel_path: Relative source path for fallback.

    Returns:
        A human-meaningful title string.
    """

    if conv_title:
        return conv_title
    meta = data.get("meta")
    if isinstance(meta, dict):
        fn = meta.get("filename")
        if isinstance(fn, str) and fn.strip():
            return Path(fn).stem
    return Path(rel_path).stem


def resolve_source_path(input_dir: str, rel: str) -> str:
    """Resolve a source path, trying with and without a trailing ``.json``.

    Parameters:
        input_dir: Root directory containing source files.
        rel: Relative path from ``input_dir`` to the source file.

    Returns:
        The resolved path string.

    Raises:
        ValueError: If neither the path nor its ``.json`` variant exists.
    """

    src_path = os.path.join(input_dir, rel)
    if os.path.isfile(src_path):
        return src_path
    alt = src_path + ".json"
    if os.path.isfile(alt):
        return alt
    raise ValueError(f"Source not found: {src_path}")


def normalize_arg_value(value: object) -> str | int | bool | None:
    """Return a serialization-friendly representation of an argument value.

    Parameters
    ----------
    value: object
        Parsed CLI argument value.

    Returns
    -------
    str | int | bool | None
        Normalized value suitable for filename serialization.
    """

    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        text_value = f"{value}"
        return text_value.rstrip("0").rstrip(".") if "." in text_value else text_value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value)


def extract_non_default_arguments(
    args: argparse.Namespace, defaults: Mapping[str, object]
) -> dict[str, Any]:
    """Return CLI arguments that differ from their defaults.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.
    defaults: Mapping[str, object]
        Mapping of argument destinations to their default values.

    Returns
    -------
    dict[str, Any]
        Mapping of argument names to normalized non-default values.
    """

    excluded = {"output_dir", "output_name", "replay_from", "resume_from"}
    params: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_") or key in excluded:
            continue
        default_value = defaults.get(key, None)
        if value == default_value:
            continue
        normalized_value = normalize_arg_value(value)
        normalized_default = normalize_arg_value(default_value)
        if normalized_value == normalized_default:
            continue
        params[key] = normalized_value
    return params
