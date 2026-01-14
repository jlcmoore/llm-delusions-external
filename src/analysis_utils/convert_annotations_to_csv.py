"""Convert annotation JSONL outputs to flat CSV tables.

This helper scans selected annotation output roots for JSONL files that match
fixed patterns and writes per-file CSVs with one row per record and a union
of all keys as columns.
"""

import csv
import json
from pathlib import Path

roots = [
    Path("annotation_outputs/human_line"),
    Path("annotation_outputs/under_irb"),
]

files = sorted(p for root in roots for p in root.rglob("20251215*.jsonl"))

for jsonl_path in files:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # drop meta rows; keep only annotation rows
            if obj.get("type") == "meta":
                continue

            rows.append(obj)

    if not rows:
        continue

    fieldnames = sorted({k for r in rows for k in r.keys()})
    csv_path = jsonl_path.with_suffix(".csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
