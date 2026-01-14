"""
Generate paraphrased variants of subset transcripts using LiteLLM.

This script reads JSON subset files under an input directory such as
``subsets/`` and writes structurally identical JSON files under a separate
output directory, with message ``content`` fields replaced by paraphrased
text. Metadata is preserved, and additional fields indicate paraphrasing
configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tqdm import tqdm

from llm_utils import ParaphraseError, paraphrase_block
from llm_utils.client import DEFAULT_CHAT_MODEL
from utils.cli import (
    add_log_level_argument,
    add_model_argument,
    add_output_path_argument,
    add_participants_argument,
    add_subset_input_argument,
)
from utils.schema import (
    PARAPHRASE_INFO_GENERATED_UTC,
    PARAPHRASE_INFO_KEY,
    PARAPHRASE_INFO_MAX_TOKENS,
    PARAPHRASE_INFO_MODEL,
    PARAPHRASE_INFO_TEMPERATURE,
    SUBSET_INFO_KEY,
    SUBSET_INFO_PARAPHRASE_TOTAL_VARIANTS,
    SUBSET_INFO_PARAPHRASE_VARIANT_INDEX,
    SUBSET_INFO_PARAPHRASED,
    SUBSET_MESSAGES_KEY,
)

DEFAULT_INPUT_DIR = "subsets"
DEFAULT_OUTPUT_DIR = "subsets_rephrase"

logger = logging.getLogger(__name__)


@dataclass
class ParaphraseConfig:
    """Configuration for paraphrasing subset transcripts."""

    input_dir: Path
    output_dir: Path
    model: str
    temperature: float
    num_variants: int
    max_tokens: Optional[int]
    timeout: Optional[int]
    overwrite: bool
    log_level: str
    participants: Optional[Set[str]]
    max_workers: int
    include_non_user_messages: bool


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the command line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Paraphrase subset transcript JSON files and write copies under a "
            "separate output directory."
        )
    )

    add_subset_input_argument(
        parser,
        flag="--input",
        default_input_dir=DEFAULT_INPUT_DIR,
    )
    add_output_path_argument(
        parser,
        default_path=DEFAULT_OUTPUT_DIR,
        help_text=(
            "Directory in which to write paraphrased subset JSON files "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    add_model_argument(parser, default_model=DEFAULT_CHAT_MODEL)
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1,
        help="Sampling temperature used when generating paraphrases.",
    )
    parser.add_argument(
        "--num-paraphrases",
        "-n",
        type=int,
        default=1,
        help=(
            "Number of paraphrased variants to generate for each input file. "
            "When greater than 1, output filenames receive a '__pN' suffix."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Optional maximum number of tokens per paraphrased message. "
            "If omitted, the model default is used."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional request timeout for each LiteLLM call in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory instead of skipping.",
    )
    add_participants_argument(
        parser,
        help_text=(
            "Restrict processing to subset files whose 'subset_info.participant' "
            "matches one of these participant IDs. May be repeated. Defaults to "
            "all participants when omitted."
        ),
    )
    parser.add_argument(
        "--include-non-user-messages",
        action="store_true",
        help=(
            "Paraphrase all message roles (assistant, tool, system, etc.) instead "
            "of only user messages. By default, only user messages are paraphrased."
        ),
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=4,
        help=(
            "Maximum number of files to process in parallel. Increase to speed up "
            "paraphrasing if your API and machine can handle more concurrent calls."
        ),
    )
    add_log_level_argument(parser)
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> ParaphraseConfig:
    """Parse command line arguments into a ``ParaphraseConfig``."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    participants: Optional[Set[str]] = None
    if args.participants:
        participants = {str(participant) for participant in args.participants}

    max_workers = int(args.max_workers)
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    config = ParaphraseConfig(
        input_dir=args.input,
        output_dir=args.output,
        model=args.model,
        temperature=float(args.temperature),
        num_variants=int(args.num_paraphrases),
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        overwrite=bool(args.overwrite),
        log_level=str(args.log_level),
        participants=participants,
        max_workers=max_workers,
        include_non_user_messages=bool(args.include_non_user_messages),
    )
    return config


def _iter_subset_files(root: Path) -> Iterable[Path]:
    """Yield all JSON subset files under ``root``."""

    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON content from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON content to ``path`` with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_output_path(
    input_root: Path,
    output_root: Path,
    input_path: Path,
    *,
    variant_index: int,
    num_variants: int,
) -> Path:
    """Return the output path for a given input file and variant index."""

    relative = input_path.relative_to(input_root)
    base = relative.with_suffix("")
    if num_variants <= 1:
        return output_root / relative

    suffix = input_path.suffix
    return output_root / base.with_name(
        f"{base.name}__p{variant_index + 1}"
    ).with_suffix(suffix)


def _paraphrase_messages_for_variant(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: Optional[int],
    include_non_user_messages: bool,
) -> List[Dict[str, Any]]:
    """Return a paraphrased copy of the messages list."""

    paraphrased: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        original_content = message.get("content", "")

        if not isinstance(original_content, str):
            paraphrased.append(dict(message))
            continue

        if not include_non_user_messages and role != "user":
            paraphrased.append(dict(message))
            continue

        variants = paraphrase_block(
            original_content,
            model=model,
            temperature=temperature,
            num_variants=1,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        new_message = dict(message)
        new_message["content"] = variants[0] if variants else original_content
        paraphrased.append(new_message)
    return paraphrased


def _augment_metadata(
    payload: Dict[str, Any],
    *,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    variant_index: int,
    num_variants: int,
) -> Dict[str, Any]:
    """Return a shallow copy of ``payload`` with paraphrase metadata added."""

    updated = dict(payload)
    subset_info = dict(updated.get(SUBSET_INFO_KEY, {}))
    subset_info[SUBSET_INFO_PARAPHRASED] = True
    subset_info[SUBSET_INFO_PARAPHRASE_VARIANT_INDEX] = variant_index + 1
    subset_info[SUBSET_INFO_PARAPHRASE_TOTAL_VARIANTS] = num_variants
    updated[SUBSET_INFO_KEY] = subset_info

    paraphrase_meta = {
        PARAPHRASE_INFO_MODEL: model,
        PARAPHRASE_INFO_TEMPERATURE: temperature,
        PARAPHRASE_INFO_MAX_TOKENS: max_tokens,
        PARAPHRASE_INFO_GENERATED_UTC: datetime.now(timezone.utc).isoformat(),
    }
    updated[PARAPHRASE_INFO_KEY] = paraphrase_meta
    return updated


def _process_single_file(config: ParaphraseConfig, input_path: Path) -> None:
    """Paraphrase a single subset JSON file according to ``config``."""

    logger.debug("Processing %s", input_path)
    try:
        payload = _load_json(input_path)
    except json.JSONDecodeError as err:
        logger.error("Failed to parse JSON file %s: %s", input_path, err)
        return

    messages = payload.get(SUBSET_MESSAGES_KEY)
    if not isinstance(messages, list):
        logger.warning("Skipping %s (missing 'messages' list).", input_path)
        return

    for variant_index in range(config.num_variants):
        output_path = _build_output_path(
            config.input_dir,
            config.output_dir,
            input_path,
            variant_index=variant_index,
            num_variants=config.num_variants,
        )
        if output_path.exists() and not config.overwrite:
            logger.info("Skipping existing file %s", output_path)
            continue

        try:
            new_messages = _paraphrase_messages_for_variant(
                messages,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                include_non_user_messages=config.include_non_user_messages,
            )
        except ParaphraseError as err:
            logger.error(
                "Paraphrasing failed for %s (variant %d): %s",
                input_path,
                variant_index,
                err,
            )
            continue

        updated_payload = _augment_metadata(
            payload,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            variant_index=variant_index,
            num_variants=config.num_variants,
        )
        updated_payload[SUBSET_MESSAGES_KEY] = new_messages
        _write_json(output_path, updated_payload)
        logger.info("Wrote paraphrased file %s", output_path)


def paraphrase_subsets(config: ParaphraseConfig) -> None:
    """Run paraphrasing over all subset JSON files using ``config``."""

    logger.info("Scanning input directory: %s", config.input_dir)
    all_files = list(_iter_subset_files(config.input_dir))
    if not all_files:
        logger.warning("No JSON files found under %s", config.input_dir)
        return

    # When a participant filter is provided, restrict the file list up front so
    # progress reporting reflects the actual number of files to process.
    if config.participants is not None:
        filtered_files: list[Path] = []
        for path in all_files:
            try:
                payload = _load_json(path)
            except json.JSONDecodeError as err:
                logger.error("Failed to parse JSON file %s: %s", path, err)
                continue
            subset_info = payload.get(SUBSET_INFO_KEY) or {}
            participant_id = subset_info.get("participant")
            if participant_id in config.participants:
                filtered_files.append(path)
            else:
                logger.debug(
                    "Skipping %s (participant %s not selected in prefilter).",
                    path,
                    participant_id,
                )
        files = filtered_files
    else:
        files = all_files

    if not files:
        logger.warning(
            "No JSON files under %s matched the requested participants.",
            config.input_dir,
        )
        return

    logger.info("Found %d JSON files to process.", len(files))

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(_process_single_file, config, path) for path in files
        ]
        with tqdm(total=len(futures), desc="Paraphrasing subset files") as progress:
            for future in as_completed(futures):
                # Let worker handle logging; just advance progress bar here.
                future.result()
                progress.update(1)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Entry point for the paraphrase_subsets command line tool."""

    config = parse_args(argv)
    numeric_level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence noisy LiteLLM internal logging unless explicitly enabled.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.disabled = True
    logger.setLevel(numeric_level)
    paraphrase_subsets(config)


if __name__ == "__main__":
    main()
