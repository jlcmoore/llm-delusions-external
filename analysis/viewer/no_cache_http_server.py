"""HTTP server for the classification viewer with simple JSON APIs.

This augments a no-cache static file server with a few endpoints that the
viewer can call locally. It intentionally does not expose any endpoint that can
trigger new LLM classifications and is intended only for inspecting existing
outputs during local development (see ``make viewer``).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
import sys
import warnings
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Final, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow.parquet as pq

from analysis_utils.annotation_tables import (
    LOCATION_WITH_CONTEXT_COLUMNS,
    build_content_mapping_for_locations,
)
from annotation.annotation_prompts import ANNOTATIONS_FILE, BASE_SCOPE_TEXT
from annotation.configs import LLM_SCORE_CUTOFF
from annotation.io import iter_annotation_output_runs
from chat.chat_utils import compute_previous_indices_skipping_roles
from utils.io import (
    get_default_transcripts_root,
    get_simplified_messages,
    iter_jsonl_dicts,
)
from utils.param_strings import string_to_dict

# Local imports are inside handlers to keep import costs low for static traffic.

CACHE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheRequestHandler(SimpleHTTPRequestHandler):
    """Serve files and handle small JSON API calls for the viewer."""

    def end_headers(self) -> None:
        for header, value in CACHE_HEADERS.items():
            self.send_header(header, value)
        super().end_headers()

    # -------- Utilities --------
    def _send_json(self, payload: object, status: int = 200) -> None:
        """Write a JSON response with the given HTTP status code.

        Parameters
        ----------
        payload: object
            JSON-serializable value to write.
        status: int
            HTTP status code (defaults to 200).
        """

        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as err:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"Failed to encode JSON: {err}"}).encode("utf-8")
            )
            return

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        """Parse and return the JSON body of a request.

        Returns
        -------
        dict
            Parsed JSON object.
        """

        length_str = self.headers.get("Content-Length", "0")
        try:
            length = int(length_str)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON: {err}") from err
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _iter_label_objects(self, path: Path) -> Iterable[dict]:
        """Yield JSON label dicts from a newline-delimited JSONL file."""

        try:
            yield from iter_jsonl_dicts(path)
        except OSError as err:
            raise OSError(f"Failed to read labels: {err}") from err

    # -------- Routing --------
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        """Serve GET for static assets and small JSON endpoints."""

        if self.path == "/api/classify-defaults":
            self._handle_get_classify_defaults()
            return
        if self.path == "/api/config":
            self._handle_get_config()
            return
        if self.path.startswith("/api/classify-metadata"):
            self._handle_get_classify_metadata()
            return
        if self.path.startswith("/api/classify-records"):
            self._handle_get_classify_records()
            return
        if self.path == "/api/classify-datasets":
            self._handle_get_classify_datasets()
            return
        if self.path == "/api/agreement-datasets":
            self._handle_get_agreement_datasets()
            return
        if self.path == "/api/llm-runs":
            self._handle_get_llm_runs()
            return
        if self.path.startswith("/api/manual-labels"):
            self._handle_get_manual_labels()
            return
        if self.path == "/api/manual-datasets":
            self._handle_get_manual_datasets()
            return
        if self.path == "/api/manual-instructions":
            self._handle_get_manual_instructions()
            return
        super().do_GET()
        return

    def do_post(self) -> None:
        """Serve POST endpoints for annotations and classification.

        Note: The HTTP server expects a method named ``do_POST``. To satisfy
        linting rules for snake_case while maintaining compatibility, an alias
        to this method named ``do_POST`` is installed at module import time.
        """

        if self.path == "/api/context-messages":
            self._handle_context_messages()
            return
        if self.path == "/api/save-manual-labels":
            self._handle_save_manual_labels()
            return
        self.send_error(404, "Unknown endpoint")

    # -------- Endpoint impls --------
    def _handle_get_config(self) -> None:
        """Return small configuration values used by viewers."""

        self._send_json({"llm_score_cutoff": LLM_SCORE_CUTOFF})

    def _handle_get_classify_defaults(self) -> None:
        """Return default parameter values for classify_chats."""

        try:
            cc = load_classify_chats()
            # Ask classify_chats for defaults by parsing with minimal required args
            # so argparse does not error on --input.
            args = cc.parse_args(["--input", ".", "--dry-run"])
            defaults = getattr(args, "_defaults", {})

            keys = [
                "model",
                "timeout",
                "follow_links",
                "prefilter_conversations",
                "max_messages",
                "randomize",
                "randomize_per_ppt",
                "randomize_conversations",
                "max_conversations",
                "reverse_conversations",
                "preceding_context",
            ]
            payload = {key: defaults.get(key) for key in keys}
            self._send_json({"defaults": payload})
        except (ImportError, OSError, ValueError, TypeError) as err:
            # Surface a concise message to the UI; details go to stderr.
            print(f"[server] classify-defaults error: {err}", file=sys.stderr)
            self._send_json({"error": str(err)}, status=500)

    def _handle_get_llm_runs(self) -> None:
        """Return a summary of available LLM classification runs."""

        root = Path("annotation_outputs")
        if not root.exists():
            self._send_json({"runs": []})
            return

        try:
            runs = list(iter_annotation_output_runs(root))
        except OSError as err:
            self._send_json({"error": str(err)}, status=500)
            return

        payload: List[dict[str, object]] = []
        for run in runs:
            payload.append(
                {
                    "path": str(run.rel_path).replace("\\", "/"),
                    "model": run.model,
                    "participants": list(run.participants),
                    "annotation_ids": list(run.annotation_ids),
                    "preceding_context": run.preceding_context,
                    "generated_at": run.generated_at,
                    "bucket": run.bucket,
                    "participant_dir": run.participant_dir,
                }
            )

            self._send_json({"runs": payload})

    def _handle_get_classify_datasets(self) -> None:
        """Return available classification datasets backed by Parquet tables."""

        root = Path("annotations")
        if not root.exists() or not root.is_dir():
            self._send_json({"datasets": []})
            return

        payload: List[dict[str, object]] = []
        for path in sorted(root.glob("*__preprocessed.parquet")):
            stem = path.stem
            key, _, _ = stem.partition("__preprocessed")
            key = key or stem
            try:
                rel_pre = path.relative_to(Path("."))
            except ValueError:
                rel_pre = path
            preprocessed_rel = str(rel_pre).replace("\\", "/")

            matches_path = path.with_name(f"{key}__matches.parquet")
            if matches_path.exists():
                try:
                    rel_matches = matches_path.relative_to(Path("."))
                except ValueError:
                    rel_matches = matches_path
                matches_rel = str(rel_matches).replace("\\", "/")
            else:
                matches_rel = ""

            filename = f"{key}.jsonl"
            label_timestamp = self._format_manual_dataset_timestamp(
                None,
                filename,
            )
            label_params = self._format_manual_dataset_params(filename)
            label_parts: List[str] = []
            if label_timestamp:
                label_parts.append(label_timestamp)
            if label_params:
                label_parts.append(label_params)
            pretty_label = " - ".join(label_parts) if label_parts else key

            payload.append(
                {
                    "key": key,
                    "label": pretty_label,
                    "preprocessed_path": preprocessed_rel,
                    "matches_path": matches_rel,
                }
            )

        self._send_json({"datasets": payload})

    def _handle_get_classify_metadata(self) -> None:
        """Return participants, annotation ids, and cutoff info for a dataset."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        key_raw = (params.get("dataset") or params.get("key") or [""])[0].strip()
        if not key_raw:
            self._send_json({"error": "Missing dataset key"}, status=400)
            return
        if "/" in key_raw or "\\" in key_raw:
            self._send_json({"error": "Invalid dataset key"}, status=400)
            return

        root = Path("annotations")
        preprocessed_path = root / f"{key_raw}__preprocessed.parquet"
        matches_path = root / f"{key_raw}__matches.parquet"
        if not preprocessed_path.exists() and not matches_path.exists():
            self._send_json({"error": "Dataset not found"}, status=404)
            return

        participants: List[str] = []
        annotation_ids: List[str] = []

        if preprocessed_path.exists():
            try:
                pf = pq.ParquetFile(preprocessed_path)
                col_names = list(pf.schema.names)
                score_cols = [
                    name
                    for name in col_names
                    if name.startswith("score__") and len(name) > len("score__")
                ]
                annotation_ids = sorted(
                    {name[len("score__") :] for name in score_cols},
                )
                if "participant" in col_names:
                    table = pf.read(columns=["participant"])
                    raw_values = table.column("participant").to_pylist()
                    participants = sorted(
                        {
                            str(value).strip()
                            for value in raw_values
                            if value not in (None, "")
                        }
                    )
            except (OSError, ValueError) as err:
                self._send_json(
                    {"error": f"Failed to inspect dataset metadata: {err}"},
                    status=500,
                )
                return

        has_matches = matches_path.exists()
        cutoffs_by_annotation: dict[str, List[int]] = {}
        if has_matches:
            try:
                pf_matches = pq.ParquetFile(matches_path)
                match_names = list(pf_matches.schema.names)
                if "score_cutoff" in match_names:
                    table = pf_matches.read(columns=["annotation_id", "score_cutoff"])
                    annot_values = table.column("annotation_id").to_pylist()
                    cutoff_values = table.column("score_cutoff").to_pylist()
                    tmp: dict[str, set[int]] = {}
                    for annot, cutoff in zip(annot_values, cutoff_values):
                        if annot is None or cutoff is None:
                            continue
                        annot_str = str(annot).strip()
                        if not annot_str:
                            continue
                        try:
                            cutoff_int = int(cutoff)
                        except (TypeError, ValueError):
                            continue
                        if annot_str not in tmp:
                            tmp[annot_str] = set()
                        tmp[annot_str].add(cutoff_int)
                    cutoffs_by_annotation = {
                        annot: sorted(values) for annot, values in tmp.items()
                    }
            except (OSError, ValueError) as err:
                warnings.warn(
                    f"Failed to inspect matches parquet for {key_raw}: {err}",
                )

        payload = {
            "key": key_raw,
            "participants": participants,
            "annotation_ids": annotation_ids,
            "has_matches": has_matches,
            "cutoffs_by_annotation": cutoffs_by_annotation,
        }
        self._send_json(payload)

    def _handle_get_classify_records(self) -> None:
        """Return a page of classification records for a dataset."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        key_raw = (params.get("dataset") or params.get("key") or [""])[0].strip()
        annotation_id = (params.get("annotation_id") or [""])[0].strip()
        participant = (params.get("participant") or ["__all__"])[0].strip()
        page_raw = (params.get("page") or ["1"])[0].strip()
        page_size_raw = (params.get("page_size") or ["25"])[0].strip()
        cutoff_raw = (params.get("score_cutoff") or [""])[0].strip()
        mode_raw = (params.get("score_mode") or [""])[0].strip().lower()

        if not key_raw:
            self._send_json({"error": "Missing dataset key"}, status=400)
            return
        if not annotation_id:
            self._send_json({"error": "Missing annotation_id"}, status=400)
            return
        if "/" in key_raw or "\\" in key_raw:
            self._send_json({"error": "Invalid dataset key"}, status=400)
            return

        try:
            page = int(page_raw)
        except ValueError:
            page = 1
        page = max(page, 1)

        try:
            page_size = int(page_size_raw)
        except ValueError:
            page_size = 25
        page_size = max(page_size, 1)
        page_size = min(page_size, 500)

        score_mode = "eq" if mode_raw == "eq" else "ge"

        root = Path("annotations")
        matches_path = root / f"{key_raw}__matches.parquet"
        preprocessed_path = root / f"{key_raw}__preprocessed.parquet"
        if not preprocessed_path.exists():
            self._send_json({"error": "Dataset not found"}, status=404)
            return

        try:
            # Always load records from the preprocessed per-message table.
            pf = pq.ParquetFile(preprocessed_path)
            col_names = list(pf.schema.names)
            score_column = f"score__{annotation_id}"
            if score_column not in col_names:
                self._send_json(
                    {"error": f"Annotation {annotation_id!r} not present in dataset"},
                    status=400,
                )
                return
            filters: List[tuple[str, str, object]] = []
            if participant and participant != "__all__":
                filters.append(("participant", "=", participant))
            frame = pd.read_parquet(
                preprocessed_path,
                columns=LOCATION_WITH_CONTEXT_COLUMNS + [score_column],
                engine="pyarrow",
                filters=filters or None,
            )
            frame.rename(columns={score_column: "score"}, inplace=True)
            score_cutoff_value: Optional[int]
            if cutoff_raw:
                try:
                    score_cutoff_value = int(cutoff_raw)
                except ValueError:
                    score_cutoff_value = None
            else:
                score_cutoff_value = None
            if score_cutoff_value is not None:
                if score_mode == "eq":
                    frame = frame[frame["score"] == score_cutoff_value]
                else:
                    frame = frame[frame["score"] >= score_cutoff_value]
            frame.insert(0, "annotation_id", annotation_id)
            # No matches information is available in the preprocessed table.
            frame["matches"] = [[] for _ in range(len(frame.index))]
            frame["content"] = ""
            print(
                f"[viewer] classify-records key={key_raw!r} "
                f"annotation_id={annotation_id!r} participant={participant!r} "
                f"cutoff_raw={cutoff_raw!r} mode={score_mode!r} "
                f"rows_after_filters={len(frame.index)}",
            )
        except (OSError, ValueError) as err:
            self._send_json(
                {"error": f"Failed to load records for dataset {key_raw}: {err}"},
                status=500,
            )
            return

        if frame.empty:
            self._send_json(
                {
                    "key": key_raw,
                    "annotation_id": annotation_id,
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "records": [],
                }
            )
            return

        frame.sort_values(
            by=["participant", "source_path", "chat_index", "message_index"],
            inplace=True,
        )

        total = int(frame.shape[0])
        start = (page - 1) * page_size
        if start >= total:
            start = max(0, total - page_size)
        end = min(start + page_size, total)
        page_frame = frame.iloc[start:end].copy()

        # Build a small enrichment mapping from transcripts_data/transcripts.parquet
        # for just the locations on this page so that per-page content loading
        # remains fast.
        enrichment: dict[tuple[str, str, int, int], dict] = {}

        # Prepare filters for looking up any available matches for this
        # annotation/participant combination. The matches parquet is used only
        # to attach validated quote spans and does not drive pagination or
        # score filtering.
        base_filters: List[tuple[str, str, object]] = [
            ("annotation_id", "=", annotation_id)
        ]
        if participant and participant != "__all__":
            base_filters.append(("participant", "=", participant))
        cutoff_for_match: Optional[int] = score_cutoff_value
        if cutoff_for_match is not None:
            filters_for_match = base_filters + [("score_cutoff", "=", cutoff_for_match)]
        else:
            filters_for_match = base_filters

        transcripts_root = Path("transcripts_data")
        transcripts_path = transcripts_root / "transcripts.parquet"
        if transcripts_path.exists() and not page_frame.empty:
            try:
                content_by_key = build_content_mapping_for_locations(
                    transcripts_path,
                    page_frame.to_dict(orient="records"),
                )
                for key_loc, content_value in content_by_key.items():
                    enrichment[key_loc] = {"content": content_value}
            except (OSError, ValueError, TypeError):
                # Best-effort enrichment; fall back to preprocessed rows when
                # transcripts parquet lookups fail.
                enrichment = {}

        if matches_path.exists() and not page_frame.empty:
            try:
                m_frame = pd.read_parquet(
                    matches_path,
                    columns=[
                        "annotation_id",
                        "participant",
                        "source_path",
                        "chat_index",
                        "message_index",
                        "score_cutoff",
                        "matches",
                    ],
                    engine="pyarrow",
                    filters=filters_for_match or None,
                )

                # If nothing matched the cutoff-specific filter, fall back to
                # all matches for this annotation (and participant when set).
                if m_frame.empty and cutoff_for_match is not None:
                    m_frame = pd.read_parquet(
                        matches_path,
                        columns=[
                            "annotation_id",
                            "participant",
                            "source_path",
                            "chat_index",
                            "message_index",
                            "score_cutoff",
                            "matches",
                        ],
                        engine="pyarrow",
                        filters=base_filters or None,
                    )

                if not m_frame.empty:
                    # Restrict to locations that appear on this page.
                    loc_keys = {
                        (
                            str(row["participant"]),
                            str(row["source_path"]),
                            int(row["chat_index"]),
                            int(row["message_index"]),
                        )
                        for row in page_frame.to_dict(orient="records")
                    }
                    if loc_keys:
                        m_frame = m_frame[
                            m_frame.apply(
                                lambda r: (
                                    str(r["participant"]),
                                    str(r["source_path"]),
                                    int(r["chat_index"]),
                                    int(r["message_index"]),
                                )
                                in loc_keys,
                                axis=1,
                            )
                        ]
                    if not m_frame.empty:
                        # Prefer rows with highest score_cutoff for each location.
                        m_frame.sort_values(
                            by=["score_cutoff"],
                            ascending=[False],
                            inplace=True,
                        )
                        # Keep only the best row per location key.
                        m_frame = m_frame.drop_duplicates(
                            subset=[
                                "participant",
                                "source_path",
                                "chat_index",
                                "message_index",
                            ],
                            keep="first",
                        )
                        for m_row in m_frame.to_dict(orient="records"):
                            key_loc = (
                                str(m_row["participant"]),
                                str(m_row["source_path"]),
                                int(m_row["chat_index"]),
                                int(m_row["message_index"]),
                            )
                            existing = enrichment.get(key_loc, {})
                            combined = dict(existing)
                            raw_matches = m_row.get("matches") or []
                            if isinstance(raw_matches, str):
                                try:
                                    parsed_matches = json.loads(raw_matches)
                                except json.JSONDecodeError:
                                    parsed_matches = []
                                if isinstance(parsed_matches, list):
                                    matches_list = parsed_matches
                                else:
                                    matches_list = []
                            elif isinstance(raw_matches, list):
                                matches_list = raw_matches
                            else:
                                matches_list = []
                            combined["matches"] = matches_list
                            enrichment[key_loc] = combined
            except (OSError, ValueError, TypeError):
                # Best-effort enrichment; fall back to preprocessed rows
                # when matches parquet lookups fail.
                enrichment = {}

        records: List[dict] = []
        for row in page_frame.to_dict(orient="records"):
            key_loc = (
                str(row.get("participant") or ""),
                str(row.get("source_path") or ""),
                int(row.get("chat_index") or 0),
                int(row.get("message_index") or 0),
            )
            enriched = enrichment.get(key_loc, {})

            content_value = enriched.get("content") or ""
            matches_value = enriched.get("matches") or []

            record = {
                "annotation_id": row.get("annotation_id"),
                "participant": row.get("participant"),
                "source_path": row.get("source_path"),
                "chat_index": row.get("chat_index"),
                "message_index": row.get("message_index"),
                "role": row.get("role"),
                "score": row.get("score"),
                "matches": matches_value,
                "content": content_value,
                "timestamp": row.get("timestamp"),
                "chat_key": row.get("chat_key"),
                "chat_date": row.get("chat_date"),
            }
            records.append(record)

        payload = {
            "key": key_raw,
            "annotation_id": annotation_id,
            "total": total,
            "page": page,
            "page_size": page_size,
            "records": records,
        }
        self._send_json(payload)

    def _handle_get_agreement_datasets(self) -> None:
        """Return a list of available agreement datasets."""

        root = Path("analysis") / "agreement"
        if not root.exists():
            self._send_json({"datasets": []})
            return

        datasets: List[dict[str, object]] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            label_timestamp = self._format_manual_dataset_timestamp(None, name)
            label_params = self._format_manual_dataset_params(name)
            label_parts: List[str] = []
            if label_timestamp:
                label_parts.append(label_timestamp)
            if label_params:
                label_parts.append(label_params)
            if label_parts:
                pretty_label = " - ".join(label_parts)
            else:
                pretty_label = name
            try:
                rel = path.relative_to(Path("."))
            except ValueError:
                rel = path
            rel_str = str(rel).replace("\\", "/")
            datasets.append(
                {
                    "key": name,
                    "path": rel_str,
                    "label": pretty_label,
                }
            )

        self._send_json({"datasets": datasets})

    def _handle_get_manual_datasets(self) -> None:
        """Return a list of available manual annotation dataset files."""

        root = Path("manual_annotation_inputs")
        if not root.exists():
            self._send_json({"datasets": []})
            return

        datasets: List[dict[str, object]] = []
        for path in sorted(root.rglob("*.jsonl")):
            try:
                rel = path.relative_to(Path("."))
            except ValueError:
                rel = path
            rel_str = str(rel).replace("\\", "/")
            generated_at: Optional[str] = None
            try:
                first_line = path.open("r", encoding="utf-8").readline()
                meta = json.loads(first_line) if first_line else {}
                if isinstance(meta, dict):
                    value = meta.get("generated_at")
                    if isinstance(value, str) and value.strip():
                        generated_at = value.strip()
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                generated_at = None

            label_timestamp = self._format_manual_dataset_timestamp(
                generated_at, path.name
            )
            label_params = self._format_manual_dataset_params(path.name)
            label_parts: List[str] = []
            if label_timestamp:
                label_parts.append(label_timestamp)
            if label_params:
                label_parts.append(label_params)
            if label_parts:
                pretty_label = " - ".join(label_parts)
            else:
                pretty_label = path.name

            datasets.append(
                {
                    "path": rel_str,
                    "name": path.name,
                    "generated_at": generated_at,
                    "label": pretty_label,
                }
            )

        self._send_json({"datasets": datasets})

    @staticmethod
    def _format_manual_dataset_timestamp(
        generated_at: Optional[str], filename: str
    ) -> Optional[str]:
        """Return a human-readable timestamp for a manual-annotation dataset.

        Parameters
        ----------
        generated_at: Optional[str]
            Optional ISO-formatted timestamp from the dataset metadata.
        filename: str
            Dataset filename, potentially containing a timestamp prefix.

        Returns
        -------
        Optional[str]
            Formatted timestamp (YYYY-MM-DD HH:MM:SS) or None when unavailable.
        """

        if generated_at:
            try:
                dt = datetime.fromisoformat(generated_at)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        base_name = Path(filename).name
        prefix, separator, _ = base_name.partition("__")
        if separator and len(prefix) == 15:
            try:
                dt = datetime.strptime(prefix, "%Y%m%d-%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        return None

    @staticmethod
    def _format_manual_dataset_params(filename: str) -> str:
        """Return a human-readable parameter summary from a dataset filename.

        Parameters
        ----------
        filename: str
            Dataset filename including any encoded parameter fragment.

        Returns
        -------
        str
            Parameters formatted as key=value pairs separated by spaces,
            or an empty string when no parameters can be parsed.
        """

        base_name = Path(filename).name
        stem, _, _ = base_name.partition(".jsonl")
        _, separator, suffix = stem.partition("__")
        if separator and suffix:
            fragment = suffix.strip()
        else:
            fragment = stem.strip()
        if not fragment:
            return ""

        try:
            params = string_to_dict(fragment)
        except (TypeError, ValueError):
            return fragment.replace("&", " ")

        if not params:
            return fragment.replace("&", " ")

        parts = [
            f"{key}={value}"
            for key, value in sorted(params.items(), key=lambda item: str(item[0]))
        ]
        return " ".join(parts)

    def _handle_get_manual_labels(self) -> None:
        """Return any previously saved manual labels for a dataset/annotator."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        dataset_path_raw = (params.get("dataset_path") or [""])[0].strip()
        annotator_raw = (params.get("annotator_id") or [""])[0].strip()
        if not dataset_path_raw or not annotator_raw:
            self._send_json(
                {"error": "dataset_path and annotator_id are required"},
                status=400,
            )
            return

        manual_root = Path("manual_annotation_inputs").resolve()
        dataset_file = Path(dataset_path_raw)
        if not dataset_file.is_absolute():
            dataset_file = Path(".").joinpath(dataset_file).resolve()
        try:
            dataset_file.relative_to(manual_root)
        except ValueError:
            self._send_json(
                {
                    "error": "Dataset path must live under manual_annotation_inputs/ for resume."
                },
                status=400,
            )
            return

        safe_annotator = re.sub(r"[^A-Za-z0-9_-]+", "_", annotator_raw) or "anon"
        base_name = dataset_file.name
        labels_root = Path("manual_annotation_labels").resolve()
        label_path = labels_root / safe_annotator / base_name
        if not label_path.exists():
            self._send_json({"labels": []})
            return

        latest_by_id: dict[str, dict] = {}
        try:
            for obj in self._iter_label_objects(label_path):
                label_id = obj.get("id")
                label_value = obj.get("label")
                if not label_id or label_value not in {
                    "yes",
                    "no",
                    "not_correctly_formatted",
                }:
                    continue
                latest_by_id[str(label_id)] = obj
        except OSError as err:
            self._send_json({"error": f"Failed to read labels: {err}"}, status=500)
            return

        self._send_json({"labels": list(latest_by_id.values())})

    def _handle_get_manual_instructions(self) -> None:
        """Return shared human-facing instructions for manual annotation."""

        instructions = (
            "You will read one target message at a time and decide whether it "
            "satisfies the selected annotation.\n\n"
            f"{BASE_SCOPE_TEXT}\n\n"
            "Use the annotation description and examples to decide whether the "
            "message matches. When in doubt, err on the side of choosing 'No'."
        )
        self._send_json({"instructions": instructions})

    def _handle_save_manual_labels(self) -> None:
        """Persist manual annotation labels next to the source dataset."""

        try:
            body = self._read_json()
        except ValueError as err:
            self._send_json({"error": str(err)}, status=400)
            return

        dataset_path_raw = str(body.get("dataset_path") or "").strip()
        annotator_raw = str(body.get("annotator_id") or "").strip()
        labels = body.get("labels")
        if not dataset_path_raw:
            self._send_json({"error": "Missing 'dataset_path'"}, status=400)
            return
        if not annotator_raw:
            self._send_json({"error": "Missing 'annotator_id'"}, status=400)
            return
        if not isinstance(labels, list) or not labels:
            self._send_json({"error": "Missing or empty 'labels' list"}, status=400)
            return

        manual_root = Path("manual_annotation_inputs").resolve()
        dataset_file = Path(dataset_path_raw)
        try:
            if not dataset_file.is_absolute():
                dataset_file = Path(".").joinpath(dataset_file).resolve()
        except OSError as err:
            self._send_json({"error": str(err)}, status=400)
            return

        try:
            dataset_file.relative_to(manual_root)
        except ValueError:
            self._send_json(
                {
                    "error": "Dataset path must live under manual_annotation_inputs/ for autosave."
                },
                status=400,
            )
            return

        safe_annotator = re.sub(r"[^A-Za-z0-9_-]+", "_", annotator_raw) or "anon"
        base_name = dataset_file.name
        labels_root = Path("manual_annotation_labels").resolve()
        target_dir = labels_root / safe_annotator
        out_path = target_dir / base_name

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            # Merge with any existing labels so updates overwrite prior rows.
            latest_by_id: dict[str, dict] = {}
            if out_path.exists():
                try:
                    for obj in self._iter_label_objects(out_path):
                        label_id = obj.get("id")
                        label_value = obj.get("label")
                        if not label_id or label_value not in {
                            "yes",
                            "no",
                            "not_correctly_formatted",
                        }:
                            continue
                        latest_by_id[str(label_id)] = obj
                except OSError as err:
                    self._send_json(
                        {"error": f"Failed to read existing labels: {err}"}, status=500
                    )
                    return
            for item in labels:
                if not isinstance(item, dict):
                    continue
                label_id = item.get("id")
                label_value = item.get("label")
                if not label_id or label_value not in {
                    "yes",
                    "no",
                    "not_correctly_formatted",
                }:
                    continue
                latest_by_id[str(label_id)] = item
            with out_path.open("w", encoding="utf-8") as handle:
                for obj in latest_by_id.values():
                    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError as err:
            self._send_json({"error": f"Failed to save labels: {err}"}, status=500)
            return

        try:
            rel_path = out_path.resolve().relative_to(Path(".").resolve())
        except ValueError:
            rel_path = out_path

        self._send_json({"ok": True, "path": str(rel_path)})

    def _handle_update_annotation(self) -> None:
        """Persist edits for a single annotation row in-place.

        Supports updating ``name``, ``description``, ``positive-examples``, and
        ``negative-examples``. Only the matching row is modified and all other
        lines in ``annotations.csv`` remain byte-for-byte identical to avoid
        noisy diffs.
        """

        try:
            body = self._read_json()
            annot_id = str(body.get("id") or "").strip()
            name = str(body.get("name") or "").strip()
            description = str(body.get("description") or "").strip()

            # Examples may be provided as strings (newline-separated) or arrays.
            def _norm_examples(key: str) -> str:
                val = body.get(key)
                if isinstance(val, list):
                    return "\n".join([str(x).strip() for x in val if str(x).strip()])
                if val is None:
                    return ""
                return str(val).replace("\r\n", "\n").replace("\r", "\n").strip()

            pos_examples = _norm_examples("positive_examples")
            neg_examples = _norm_examples("negative_examples")
            if not annot_id:
                self._send_json({"error": "Missing 'id'"}, status=400)
                return

            annotations_path = Path(ANNOTATIONS_FILE)
            if not annotations_path.exists():
                self._send_json(
                    {"error": f"annotations.csv not found at {annotations_path}"},
                    status=500,
                )
                return

            try:
                updated = _update_single_csv_row(
                    annotations_path,
                    annot_id,
                    name or None,
                    description or None,
                    pos_examples or None,
                    neg_examples or None,
                )
            except (OSError, ValueError) as err:
                self._send_json({"error": str(err)}, status=500)
                return

            if not updated:
                self._send_json(
                    {"error": f"Annotation id {annot_id!r} not found"}, status=404
                )
                return

            self._send_json({"ok": True})
        except ValueError as err:
            self._send_json({"error": str(err)}, status=400)
        except OSError as err:
            self._send_json({"error": f"Failed to write CSV: {err}"}, status=500)

    def _handle_context_messages(self) -> None:
        """Return neighboring messages for a classified record.

        Expects a JSON body with:

        - ``source_path``: Relative path to the transcript JSON beneath
          ``transcripts_de_ided/``.
        - ``chat_index``: Zero-based conversation index used during
          classification.
        - ``message_index``: Zero-based message index within the conversation.
        - ``depth``: Optional maximum number of messages to return before and
          after the target message.
        """

        try:
            body = self._read_json()
        except ValueError as err:
            self._send_json({"error": str(err)}, status=400)
            return

        source_path = str(body.get("source_path") or "").strip()
        if not source_path:
            self._send_json({"error": "Missing 'source_path'"}, status=400)
            return

        try:
            chat_index_raw = body.get("chat_index")
            message_index_raw = body.get("message_index")
            chat_index = int(chat_index_raw)
            message_index = int(message_index_raw)
        except (TypeError, ValueError):
            self._send_json(
                {"error": "chat_index and message_index must be integers"},
                status=400,
            )
            return

        depth_raw = body.get("depth")
        try:
            depth = int(depth_raw) if depth_raw is not None else 3
        except (TypeError, ValueError):
            self._send_json({"error": "depth must be an integer"}, status=400)
            return

        try:
            previous, next_messages = load_context_messages(
                source_path, chat_index, message_index, depth
            )
        except (FileNotFoundError, OSError, ValueError) as err:
            self._send_json({"error": str(err)}, status=400)
            return

        self._send_json({"previous": previous, "next": next_messages})


def _extract_cost_info(stdout_text: str) -> dict | None:
    """Return a structured cost estimate parsed from dry-run stdout.

    Looks for JSON objects containing either ``max_total_cost_usd`` (preferred)
    or ``max_potential_cost_usd``. Returns a minimal dict with fields:
    ``maxCostUsd``, ``totalRequests``, and optional ``model``.
    """

    try:
        objects = _extract_json_objects_from_text(stdout_text)
    except (ValueError, TypeError, json.JSONDecodeError):  # defensive
        return None

    for obj in objects:
        if isinstance(obj, dict) and "max_total_cost_usd" in obj:
            return {
                "maxCostUsd": obj.get("max_total_cost_usd"),
                "totalRequests": obj.get("total_request_count"),
                "model": obj.get("model"),
            }
    for obj in objects:
        if isinstance(obj, dict) and "max_potential_cost_usd" in obj:
            return {
                "maxCostUsd": obj.get("max_potential_cost_usd"),
                "totalRequests": obj.get("estimated_request_count"),
                "model": None,
            }
    return None


def _extract_json_objects_from_text(text: str) -> list[dict]:
    """Extract JSON objects embedded in arbitrary text by tracking braces.

    This avoids brittle regex; it is limited but adequate for our known output.
    """

    objects: list[dict] = []
    depth = 0
    in_string = False
    escape = False
    start = -1
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = text[start : idx + 1]
                try:
                    obj = json.loads(chunk)
                except (
                    ValueError,
                    json.JSONDecodeError,
                    TypeError,
                ):  # ignore parse errors
                    start = -1
                    continue
                if isinstance(obj, dict):
                    objects.append(obj)
                start = -1
    return objects


def load_context_messages(
    source_path: str, chat_index: int, message_index: int, depth: int
) -> Tuple[List[dict], List[dict]]:
    """Return messages before and after a target message using chat loaders.

    This mirrors the loading performed in ``src/chat/chat_io.py`` and
    ``src/chat/chat_utils.py`` so that the viewer context matches the
    conversations used during classification.

    Parameters
    ----------
    source_path:
        Relative path recorded in the JSONL output (``source_path`` field).
    chat_index:
        Zero-based index of the conversation within the file.
    message_index:
        Zero-based index of the target message within the conversation.
    depth:
        Maximum number of messages to include before and after the target.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        Two lists of simplified message dicts: ``(previous, next)``.
    """

    transcripts_root = get_default_transcripts_root()
    messages = get_simplified_messages(transcripts_root, source_path, chat_index)
    if not messages:
        return [], []

    depth_limit = depth if depth > 0 else 0
    depth_limit = min(depth_limit, 10)

    previous_indices = compute_previous_indices_skipping_roles(
        messages,
        message_index,
        depth_limit,
        skip_roles=("tool",),
    )
    next_start = message_index + 1
    next_end = min(len(messages), next_start + depth_limit)
    next_range = range(next_start, next_end)

    previous = [messages[i] for i in previous_indices]
    next_messages = [messages[i] for i in next_range]
    return previous, next_messages


def _update_single_csv_row(
    path: Path,
    annot_id: str,
    name: str | None,
    description: str | None,
    positive_examples: str | None,
    negative_examples: str | None,
) -> bool:
    """Update only the matching CSV record in-place, preserving other lines.

    Parameters
    ----------
    path: Path
        Path to ``annotations.csv``.
    annot_id: str
        Annotation identifier to update.
    name: str | None
        Optional new ``name`` value. If ``None`` or empty, the existing value is
        left unchanged.
    description: str | None
        Optional new ``description`` value. If ``None`` or empty, the existing
        value is left unchanged.
    positive_examples: str | None
        Optional newline-separated list to write to the ``positive-examples``
        column. If ``None`` or empty, the existing value is left unchanged.
    negative_examples: str | None
        Optional newline-separated list to write to the ``negative-examples``
        column. If ``None`` or empty, the existing value is left unchanged.

    Returns
    -------
    bool
        ``True`` if a row was updated, ``False`` if the id was not found.
    """

    data = path.read_bytes()

    # Scan bytes to split into CSV records while respecting quoted newlines.
    records: list[bytes] = []
    delims: list[bytes] = []  # delimiter that followed each record
    i = 0
    start = 0
    in_quotes = False
    length = len(data)
    while i < length:
        b = data[i]
        if b == 0x22:  # '"'
            if in_quotes:
                # Escaped quote inside quoted field: ""
                if i + 1 < length and data[i + 1] == 0x22:
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            in_quotes = True
            i += 1
            continue
        if not in_quotes and b in (0x0A, 0x0D):  # LF or CR
            # End of record at i (exclusive)
            end = i
            # Determine delimiter
            if b == 0x0D and i + 1 < length and data[i + 1] == 0x0A:
                delim = b"\r\n"
                i += 2
            else:
                delim = bytes([b])
                i += 1
            records.append(data[start:end])
            delims.append(delim)
            start = i
            continue
        i += 1
    # Trailing record (no final newline)
    if start <= length:
        trailing = data[start:length]
        if trailing:
            records.append(trailing)
            delims.append(b"")

    if not records:
        raise ValueError("annotations.csv appears to be empty")

    # Parse header to find column indices
    header_text = records[0].decode("utf-8")
    header_reader = csv.reader(io.StringIO(header_text))
    header_row = next(header_reader, None) or []
    # Normalize header names for matching
    name_to_idx = {str(h): idx for idx, h in enumerate(header_row)}
    try:
        id_idx = name_to_idx["id"]
    except KeyError as err:
        raise ValueError("annotations.csv is missing 'id' column") from err
    # Only update these if present
    name_idx = name_to_idx.get("name")
    desc_idx = name_to_idx.get("description")
    pos_idx = name_to_idx.get("positive-examples")
    neg_idx = name_to_idx.get("negative-examples")

    updated = False
    updated_index = -1
    header_delim = delims[0] if delims else b"\n"
    # Walk records (skip header at index 0)
    for rec_index in range(1, len(records)):
        rec_text = records[rec_index].decode("utf-8")
        # Use csv to parse a single logical row (may include commas/quotes)
        row_reader = csv.reader(io.StringIO(rec_text))
        row = next(row_reader, None)
        if row is None:
            continue
        # Defensive: pad row to header length so index access is safe
        if len(row) < len(header_row):
            row = row + [""] * (len(header_row) - len(row))
        current_id = (row[id_idx] or "").strip()
        if current_id != annot_id:
            continue
        # Apply updates
        if name_idx is not None and name is not None:
            row[name_idx] = name
        if desc_idx is not None and description is not None:
            row[desc_idx] = description
        if pos_idx is not None and positive_examples is not None:
            row[pos_idx] = positive_examples
        if neg_idx is not None and negative_examples is not None:
            row[neg_idx] = negative_examples
        # Re-render only this record using csv with no trailing newline
        sink = io.StringIO()
        writer = csv.writer(sink, lineterminator="")
        writer.writerow(row[: len(header_row)])
        records[rec_index] = sink.getvalue().encode("utf-8")
        updated = True
        updated_index = rec_index
        break

    if not updated:
        return False

    # Stitch back together preserving original delimiters between logical rows
    out_parts: list[bytes] = []
    for idx, chunk in enumerate(records):
        out_parts.append(chunk)
        if idx < len(delims):
            # For the updated row, prefer the file's header delimiter to avoid
            # introducing stray CR characters if the file is LF-based.
            if idx == updated_index and header_delim:
                out_parts.append(header_delim)
            else:
                out_parts.append(delims[idx])
    new_data = b"".join(out_parts)
    path.write_bytes(new_data)
    return True


def parse_args() -> argparse.Namespace:
    """Return command-line arguments for HTTP server configuration."""
    parser = argparse.ArgumentParser(
        description="Start a no-cache HTTP server for development assets."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory to serve (defaults to current directory).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind against (defaults to 8000).",
    )
    return parser.parse_args()


def serve(directory: Path, port: int) -> None:
    """Start the HTTP server bound to the requested directory and port."""
    handler = partial(NoCacheRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("localhost", port), handler)

    print(
        f"Serving {directory.resolve()} on http://localhost:{port}/ with caching disabled."
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    """Entry point that starts the no-cache HTTP server."""
    # Suppress noisy multiprocessing resource_tracker warnings on Ctrl+C shutdown
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be .* leaked semaphore objects",
        category=UserWarning,
    )
    args = parse_args()
    serve(directory=args.directory, port=args.port)


# Install camel-case HTTP method alias after class definition to satisfy http.server
# while keeping a snake_case implementation for linting.
NoCacheRequestHandler.do_POST = NoCacheRequestHandler.do_post


def load_classify_chats() -> ModuleType:
    """Load the ``scripts/annotation/classify_chats.py`` module.

    Returns
    -------
    ModuleType
        Loaded module exposing ``parse_args`` and ``main``.
    """
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "scripts" / "annotation" / "classify_chats.py"
    module_name = "scripts.annotation.classify_chats"
    scripts_dir = str((project_root / "scripts").resolve())
    # Ensure the scripts package root is importable so annotation modules resolve
    added_path = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added_path = True

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to locate classify_chats at {module_path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert hasattr(loader, "exec_module")  # narrow type for mypy/pylint
    # Ensure fresh annotations on each load so edits in annotations.csv are honored
    if "annotations" in sys.modules:
        del sys.modules["annotations"]
    # Register in sys.modules so decorators (e.g., dataclasses) can resolve __module__
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    finally:
        # Leave scripts_dir on sys.path so subsequent imports behave consistently
        # (e.g., later classify runs). Do not remove to avoid race conditions.
        if added_path:
            pass
    return module


def _param_str(params: dict, key: str) -> str:
    """Return a trimmed string value or empty string when missing."""

    val = params.get(key)
    return str(val).strip() if val is not None else ""


def _param_bool(params: dict, key: str) -> bool:
    """Return a boolean flag for ``key``.

    Any truthy value is treated as ``True``; falsy as ``False``.
    """

    return bool(params.get(key))


def _param_int(params: dict, key: str) -> int | None:
    """Return an integer value or ``None`` if missing."""

    if params.get(key) is None:
        return None
    try:
        return int(params[key])
    except (TypeError, ValueError) as err:
        raise ValueError(f"{key} must be an integer") from err


def _collect_participants(params: dict) -> list[str]:
    """Normalize participant identifiers from list or comma string."""

    raw = params.get("participants")
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(p).strip() for p in raw]
    else:
        items = []
    return [p for p in items if p]


def _append_opt(argv: list[str], flag: str, value: str) -> None:
    """Append ``flag value`` when ``value`` is a non-empty string."""

    if value:
        argv.extend([flag, value])


def _append_int(argv: list[str], flag: str, value: int | None) -> None:
    """Append ``flag value`` when ``value`` is not None."""

    if value is not None:
        argv.extend([flag, str(value)])


def _append_flag(argv: list[str], flag: str, enabled: bool) -> None:
    """Append ``flag`` when ``enabled`` is True."""

    if enabled:
        argv.append(flag)


if __name__ == "__main__":
    main()
