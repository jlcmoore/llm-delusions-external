# Subsets

TODO: make sure this is up to date

## Contents

- [Creating Subsets](#creating-subsets)
- [Scoring Subsets](#scoring-subsets)
- [Paraphrasing subset transcripts](#paraphrasing-subset-transcripts)
- [Automatic subset creation](#automatic-subset-creation)

## Creating Subsets

Use the subsets plan CSV to extract small windows of messages around quoted anchors and write them to a mirrored `subsets/` tree. Ensure your virtualenv is active (`source env-delusions/bin/activate`).

Download the latest plan to `./subsets.csv` from:

- https://docs.google.com/spreadsheets/d/1eNq1_TMinJ4dzenvjxev46wtvq-MPaJgBU4qItFUbKQ/

The CSV should be fully quoted to avoid issues with curly quotes in spreadsheet exports.

Run a dry-run to validate rows without writing files:

```
python scripts/subsets/make_subsets.py --csv subsets/subsets.csv --dry-run --verbose
```

Generate subsets into `./subsets` (mirrors structure under `transcripts_de_ided`):

```
python scripts/subsets/make_subsets.py --csv subsets.csv --input-dir transcripts_de_ided --output-dir subsets --verbose
```

CSV columns (headers required):

- `rel_path` — path relative to `--input-dir` (e.g., `under_irb/irb_05/chat.html.json`).
- `conversation_id` — conversation title/key (blank if file has a single conversation).
- `quote` — substring to anchor on (case-insensitive fallback).
- `label` — one of `normal`, `pivotal`, `harmful` (used in output filename).
- `participant` — participant ID string (e.g., `irb_05`, `hl_12`).
- `prev_count` — number of messages before the match to include.
- `after_count` — number of messages after the match to include.
- `comments` — optional notes copied into the output metadata.

Output files are named `{label}_{short-title}_{short-quote}.json` and include a flat `messages` list plus a header noting they are auto-generated.

## Scoring Subsets

Subsets are scored in three stages using shared LLM infrastructure:

1. **Quality scores for each subset** (prior conversation reliance, uploaded document reliance, cohesion):

   ```bash
   python scripts/subsets/classify_subsets.py \
     --input-dir subsets \
     --output subsets/subset_quality.jsonl \
     --plan-csv subsets/subsets.csv \
     --scores-csv subsets/subset_quality_scores_raw__gpt-5.1.csv \
     --model gpt-5.1
   ```

   - Use `--dry-run` to preview prompts and estimated cost without sending any requests.

2. **Combine quality scores and harmful timing into a spreadsheet-aligned CSV**:

   ```bash
   python scripts/subsets/summarize_subset_harmfulness.py \
     --subset-quality-json subsets/subset_quality.jsonl \
     --subsets-root subsets \
     --plan-csv subsets/subsets.csv \
     --output-csv subsets/subset_quality_scores.csv \
     --early-turn-threshold 50
   ```

   - The resulting `subset_quality_scores.csv` has one row per plan row (aligned with `subsets.csv`), fully quoted for spreadsheet import.
   - It includes the three quality scores, LLM notes, harmful-annotation presence and earliest harmful turn, and a `passes_quality_filters` flag computed from all of these.

## Paraphrasing subset transcripts

For some analyses we generate paraphrased variants of subset transcripts so we can inspect meaning-equivalent but lexically distinct copies of chats.

Use `scripts/subsets/paraphrase_subsets.py` to read JSON subsets under `subsets/` and write paraphrased copies under a separate directory (by default `subsets_rephrase/`), preserving metadata:

```bash
python scripts/subsets/paraphrase_subsets.py \
  --input subsets \
  --output subsets_rephrase \
  --participant irb_05 \
  --model gpt-4.1-2025-04-14 \
  --num-paraphrases 1
```

Key behavior and flags:

- Only `messages[*].content` fields are paraphrased; the surrounding JSON structure and `subset_info` metadata are preserved, with additional `paraphrase_info` fields describing the model and settings used.
- By default, only `role == "user"` messages are paraphrased; pass `--include-non-user-messages` to also paraphrase `assistant`, `tool`, and other roles.
- Use `--participant` (repeatable) to restrict processing to specific participants (e.g. `--participant irb_05`).
- Use `--num-paraphrases N` to generate multiple paraphrased variants per subset file (written as `__p1`, `__p2`, … suffixes).
- Use `--temperature`, `--max-tokens`, and `--timeout` to tune the paraphrasing model behavior and output length.
- Use `--max-workers N` to control how many files are processed in parallel; each file is paraphrased independently with its own series of LiteLLM calls.

## Automatic subset creation

You can also bootstrap a subsets plan directly from LLM annotation outputs:

1. Aggregate conversation-level counts for a single annotation:

   ```bash
   python scripts/annotation/annotation_conversation_counts.py \
     -a user-endorses-delusion \
     --score-cutoff 8 \
     --min-occurrences 3 \
     "annotation_outputs/human_line/hl_01/20251215-203659__model=gpt-5.1&preceding_context=0&all-messages&most-ppts.jsonl"
   ```

   This writes a conversation-level CSV to `subsets/annotation_conversation_counts/`, for example:
   - `subsets/annotation_conversation_counts/annotation_id=user-endorses-delusion&min_occurrences=3&score_cutoff=8.csv`

2. Build an auto-generated subsets plan from that CSV:

   ```bash
   python scripts/subsets/build_subsets_plan_from_annotations.py \
     --conversation-counts-csv "subsets/annotation_conversation_counts/annotation_id=user-endorses-delusion&min_occurrences=3&score_cutoff=8.csv" \
     -a user-endorses-delusion \
     --label harmful \
     --prev-count 20 \
     --after-count 20 \
     "annotation_outputs/human_line/hl_01/20251215-203659__model=gpt-5.1&preceding_context=0&all-messages&most-ppts.jsonl"
   ```

   This produces an auto plan at:
   - `subsets/auto_subsets/annotation_id=user-endorses-delusion&min_occurrences=3&score_cutoff=8.csv`

   You can point `make_subsets.py` directly at this CSV, or open it in a spreadsheet and paste selected rows into your canonical `subsets.csv` for further refinement.

3. Generate subsets from the auto plan into a separate items directory:

   ```bash
   python scripts/subsets/make_subsets.py \
     --csv "subsets/auto_subsets/annotation_id=user-endorses-delusion&min_occurrences=3&score_cutoff=8.csv" \
     --input-dir transcripts_de_ided \
     --output-dir subsets/items_user-endorses-delusion_s8_min3 \
     --verbose
   ```

4. Classify those subsets for prior/uploaded/cohesion and write a run-specific quality file:

   ```bash
   python scripts/subsets/classify_subsets.py \
     --input-dir subsets/items_user-endorses-delusion_s8_min3 \
     --output subsets/subset_quality__user-endorses-delusion_s8_min3.jsonl \
     --plan-csv "subsets/auto_subsets/annotation_id=user-endorses-delusion&min_occurrences=3&score_cutoff=8.csv" \
     --scores-csv subsets/subset_quality_scores__user-endorses-delusion_s8_min3.csv \
     --model gpt-5.1
   ```

5. Summarize quality scores (and, when harmful-only annotations over these subsets are available, harmfulness flags) into a spreadsheet-aligned CSV:

   ```bash
   python scripts/subsets/summarize_subset_harmfulness.py \
     --subset-quality-json subsets/subset_quality__user-endorses-delusion_s8_min3.jsonl \
     --subsets-root subsets/items_user-endorses-delusion_s8_min3 \
     --plan-csv "subsets/auto_subsets/annotation_id=user-endorses-delusion&min_occurrences=3&score_cutoff=8.csv" \
     --output-csv subsets/subset_quality_scores__user-endorses-delusion_s8_min3.csv
   ```
