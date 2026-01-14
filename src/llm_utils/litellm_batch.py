"""
Utilities for running LiteLLM batch jobs across multiple providers.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from typing import Any, Mapping, Sequence

DEFAULT_BATCH_FILENAME = "litellm_batch.jsonl"
DEFAULT_ENDPOINT = "/v1/chat/completions"

Messages = Sequence[Mapping[str, Any]]


class BatchTimeoutError(TimeoutError):
    """Raised when a LiteLLM batch does not complete within the allotted time."""


class BatchFailedError(RuntimeError):
    """Raised when a LiteLLM batch fails on the provider side."""


def to_batch_json(
    keys_to_messages: Mapping[str, Messages],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    request_parameters: Mapping[str, Any] | None = None,
) -> bytes:
    """
    Convert a mapping of request keys to JSONL formatted LiteLLM batch payload.
    """
    if not keys_to_messages:
        return b""

    body_defaults: dict[str, Any] = {}
    if request_parameters:
        body_defaults.update(request_parameters)

    lines: list[str] = []
    for custom_id, messages in keys_to_messages.items():
        body: dict[str, Any] = {"messages": list(messages)}
        body.update(body_defaults)
        line_payload = {
            "custom_id": custom_id,
            "method": "POST",
            "url": endpoint,
            "body": body,
        }
        lines.append(json.dumps(line_payload))

    return ("\n".join(lines)).encode("utf-8")


def process_batch_results(
    content: str,
    *,
    return_all_choices: bool = False,
) -> dict[str, Any]:
    """
    Parse LiteLLM batch output content into a keyed dictionary of results.
    """
    results: dict[str, Any] = {}

    for line in content.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = payload.get("custom_id")
        if not custom_id:
            continue

        response_body = payload.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])

        if return_all_choices:
            results[custom_id] = choices
            continue

        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})
        results[custom_id] = message.get("content", "")

    return results


def create_litellm_batch(
    keys_to_messages: Mapping[str, Messages],
    *,
    litellm_client: Any,
    custom_llm_provider: str = "openai",
    endpoint: str = DEFAULT_ENDPOINT,
    completion_window: str = "24h",
    request_parameters: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Create a LiteLLM batch and return its identifiers.

    Parameters
    ----------
    keys_to_messages:
        Mapping from custom identifiers to chat message payloads.
    litellm_client:
        LiteLLM client instance used to create files and batches.
    custom_llm_provider:
        Provider identifier understood by LiteLLM (for example, "openai").
    endpoint:
        Target API endpoint for the batch, such as "/v1/chat/completions".
    completion_window:
        Provider-specific completion window string, such as "24h".
    request_parameters:
        Optional default request parameters, including the model name.

    Returns
    -------
    tuple[str, str]
        The created batch identifier and the input file identifier.
    """
    batch_bytes = to_batch_json(
        keys_to_messages,
        endpoint=endpoint,
        request_parameters=request_parameters,
    )

    file_like = (DEFAULT_BATCH_FILENAME, io.BytesIO(batch_bytes))
    file_response = litellm_client.create_file(
        file=file_like,
        purpose="batch",
        custom_llm_provider=custom_llm_provider,
    )

    file_id = getattr(file_response, "id", None)
    if not file_id:
        raise ValueError("Invalid file upload response")

    batch_response = litellm_client.create_batch(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        custom_llm_provider=custom_llm_provider,
    )

    batch_id = getattr(batch_response, "id", None)
    if not batch_id:
        raise ValueError("Invalid batch response")

    return str(batch_id), str(file_id)


def poll_litellm_batch(
    batch_id: str,
    *,
    litellm_client: Any,
    custom_llm_provider: str = "openai",
    polling_interval: float = 10.0,
    timeout_seconds: float = 25 * 60 * 60,
) -> Any:
    """
    Poll a LiteLLM batch until completion, failure, or timeout.

    Parameters
    ----------
    batch_id:
        Identifier of the batch to poll.
    litellm_client:
        LiteLLM client instance used to retrieve batch status.
    custom_llm_provider:
        Provider identifier understood by LiteLLM (for example, "openai").
    polling_interval:
        Delay in seconds between successive status checks.
    timeout_seconds:
        Maximum time in seconds to wait for completion.

    Returns
    -------
    Any
        The final retrieve response object for the batch.

    Raises
    ------
    BatchTimeoutError
        If the batch does not complete before timeout_seconds.
    BatchFailedError
        If the provider reports a failed batch.
    """
    if polling_interval <= 0:
        raise ValueError("polling_interval must be greater than zero")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    deadline = time.monotonic() + timeout_seconds
    retrieve_response: Any = None

    while True:
        retrieve_response = litellm_client.retrieve_batch(
            batch_id=batch_id,
            custom_llm_provider=custom_llm_provider,
        )

        status = getattr(retrieve_response, "status", None)
        if not status:
            raise ValueError("Invalid retrieve response")

        failed_at = getattr(retrieve_response, "failed_at", None)
        if failed_at:
            errors = getattr(retrieve_response, "errors", "Batch failed")
            raise BatchFailedError(str(errors))

        if status == "completed":
            return retrieve_response

        if time.monotonic() >= deadline:
            raise BatchTimeoutError(
                f"Batch {batch_id} did not complete within {timeout_seconds} seconds"
            )

        time.sleep(polling_interval)


def download_litellm_batch_output(
    output_file_id: str,
    *,
    litellm_client: Any,
    custom_llm_provider: str = "openai",
) -> str:
    """
    Download and decode the output file for a completed LiteLLM batch.

    Parameters
    ----------
    output_file_id:
        Identifier of the output file produced by the batch.
    litellm_client:
        LiteLLM client instance used to fetch file content.
    custom_llm_provider:
        Provider identifier understood by LiteLLM (for example, "openai").

    Returns
    -------
    str
        Decoded UTF-8 content of the batch output file.
    """
    if not output_file_id:
        raise ValueError("Batch completed without an output file")

    content_response = litellm_client.file_content(
        file_id=output_file_id,
        custom_llm_provider=custom_llm_provider,
    )
    binary_content = getattr(content_response, "content", None)
    if binary_content is None:
        raise ValueError("Missing content in file response")

    return binary_content.decode("utf-8")


def delete_litellm_file(
    file_id: str,
    *,
    litellm_client: Any,
    custom_llm_provider: str = "openai",
) -> None:
    """
    Delete a LiteLLM file resource, suppressing attribute errors.

    Parameters
    ----------
    file_id:
        Identifier of the file to delete.
    litellm_client:
        LiteLLM client instance used to delete files.
    custom_llm_provider:
        Provider identifier understood by LiteLLM (for example, "openai").
    """
    if not file_id:
        return
    with contextlib.suppress(AttributeError):
        litellm_client.file_delete(
            file_id=file_id,
            custom_llm_provider=custom_llm_provider,
        )


def resume_litellm_batch(
    batch_id: str,
    *,
    litellm_client: Any,
    custom_llm_provider: str = "openai",
    polling_interval: float = 10.0,
    timeout_seconds: float = 25 * 60 * 60,
    return_all_choices: bool = False,
) -> dict[str, Any]:
    """
    Resume a previously created LiteLLM batch and return parsed results.

    Parameters
    ----------
    batch_id:
        Identifier of the existing batch to resume.
    litellm_client:
        LiteLLM client instance used to poll and download results.
    custom_llm_provider:
        Provider identifier understood by LiteLLM (for example, "openai").
    polling_interval:
        Delay in seconds between successive status checks.
    timeout_seconds:
        Maximum time in seconds to wait for completion.
    return_all_choices:
        When True, return raw choices per custom identifier instead of content.

    Returns
    -------
    dict[str, Any]
        Mapping from custom identifiers to parsed result payloads.
    """
    retrieve_response = poll_litellm_batch(
        batch_id,
        litellm_client=litellm_client,
        custom_llm_provider=custom_llm_provider,
        polling_interval=polling_interval,
        timeout_seconds=timeout_seconds,
    )

    output_file_id = getattr(retrieve_response, "output_file_id", None)
    content = download_litellm_batch_output(
        str(output_file_id or ""),
        litellm_client=litellm_client,
        custom_llm_provider=custom_llm_provider,
    )
    return process_batch_results(
        content,
        return_all_choices=return_all_choices,
    )


def run_litellm_batch(
    keys_to_messages: Mapping[str, Messages],
    *,
    litellm_client: Any,
    custom_llm_provider: str = "openai",
    endpoint: str = DEFAULT_ENDPOINT,
    completion_window: str = "24h",
    polling_interval: float = 10.0,
    timeout_seconds: float = 25 * 60 * 60,
    request_parameters: Mapping[str, Any] | None = None,
    return_all_choices: bool = False,
) -> dict[str, Any]:
    """
    Execute a LiteLLM batch request and return parsed results.
    """
    batch_id: str | None = None
    file_id: str | None = None
    try:
        batch_id, file_id = create_litellm_batch(
            keys_to_messages,
            litellm_client=litellm_client,
            custom_llm_provider=custom_llm_provider,
            endpoint=endpoint,
            completion_window=completion_window,
            request_parameters=request_parameters,
        )
        retrieve_response = poll_litellm_batch(
            batch_id,
            litellm_client=litellm_client,
            custom_llm_provider=custom_llm_provider,
            polling_interval=polling_interval,
            timeout_seconds=timeout_seconds,
        )
        output_file_id = getattr(retrieve_response, "output_file_id", None)
        content = download_litellm_batch_output(
            str(output_file_id or ""),
            litellm_client=litellm_client,
            custom_llm_provider=custom_llm_provider,
        )
        return process_batch_results(
            content,
            return_all_choices=return_all_choices,
        )
    finally:
        if file_id is not None:
            delete_litellm_file(
                file_id,
                litellm_client=litellm_client,
                custom_llm_provider=custom_llm_provider,
            )
