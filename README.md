# LLM Delusions

## Contents

- [Install](#install)
- [Contributing](#contributing)
- [Parsing](#parsing)
- [Annotation](#annotation)
- [Subsets](#subsets)
- [Analyses](#analyses)
- [Participant Naming Conventions](#participant-naming-conventions)
- [Repo Structure](#repo-structure)

## Install

### Dependencies

- `python>=3.11`
- `make`
- `git-lfs` (e.g. `brew install git-lfs && git lfs install`)
- `node` and `npm`

### Set-up

`make init`

This bootstraps the Python virtual environment and installs the JavaScript dependencies with npm.

### De-ided Transcripts

These are available by default (make sure you have `git-lfs` installed) here: `./transcripts_de_ided/`.

### (Optional) Downloading transripts with identifiers

You need a local copy of the original transcripts (in `./transcripts`) in order to run a variety of the utilities in [Methods](#Methods). They are available [here](https://drive.google.com/drive/folders/1KlzUuEswIONxazVuq5JxLxyfO7r5-iaA) if you are active on the IRB.

### (Optional) Downloading Up-to-date Survey Responses

Download an up to date version of the [metadata file](https://docs.google.com/spreadsheets/d/1apQTDmkLQhl10auQASk_6QI2JsneNvMQScHLt4UeAyk/) to `transcripts/metadata.csv` and download the [Qualtrics responses](https://stanforduniversity.qualtrics.com/responses/#/surveys/SV_9Fw34krqw8QACV0) to `./qualtrics.csv`. The metadata file should have a linking between Qualtrics response IDs and our internal IDs. (This needs to be populated manually.) Also download the subsets plan spreadsheet to `./subsets.csv` from this Google Sheet: https://docs.google.com/spreadsheets/d/1eNq1_TMinJ4dzenvjxev46wtvq-MPaJgBU4qItFUbKQ/.

Then, to populate the survey responses in `surveys`, run:

```
python scripts/misc/qualtrics_to_json.py
```

Be sure to read through all of the responses before committing changes to remove any PII.

### (Optional) Downloading Up-to-date Annotation Criteria

Download an up to date version of the [annotations file](https://docs.google.com/spreadsheets/d/1zw1_FUHEkx0PZ-UH-gWZRHPlcIjA8ATnG9z3zWBfDow) to `src/data/annotations.csv`.

### (Optional) Google Cloud Storage credentials

You probably don't need to set up Google Cloud credentials. If you need them ask Jared for credentials. With those credentials, you will:

1. Install the Google Cloud CLI (`gcloud`) and log in:

```bash
gcloud auth login
```

2. (Recommended for programmatic access) Set up Application Default Credentials so local tools and Python clients can use your account:

```bash
gcloud auth application-default login
```

### (Optional) Uploading annotation outputs to Google Cloud Storage

To keep large annotation outputs out of Git (and Git LFS), sync the local `annotation_outputs/` directory to a GCS bucket. Assuming you have created a bucket called `llm-delusions-annotations` and completed the `gcloud` authentication steps above, run:

```bash
gcloud storage rsync --recursive annotation_outputs gs://llm-delusions-annotations/annotation_outputs/
```

This performs a one-way sync from your local `annotation_outputs/` tree into the bucket path `gs://llm-delusions-annotations/annotation_outputs/`, creating or updating objects as needed.

## Contributing

Please comment, lint (`make pylint`), and format (`make pyfmt` or `make jslint` or `make mdfmt`) your code. If we start using other languages, we should add a linter and formatter for it.

For major changes please also submit a pull request.

## Parsing

See [README_Parse.md](README_Parse.md) for the full parsing and de-identification pipeline for raw transcripts.

## Annotation

See [README_Annotation.md](README_Annotation.md) for how to run LLM-based classification, review outputs, and compute agreement and summary statistics.

## Subsets

See [README_Subsets.md](README_Subsets.md) for creating, scoring, and paraphrasing focused subsets of transcripts for downstream analysis.

## Analyses

See [README_Analyses.md](README_Analyses.md) for scripts that build transcript and annotation tables and run the main quantitative analyses used in the paper.

## Participant Naming Conventions

- `irb_XX` — IRB-protected participants where `XX` is a zero-padded number (e.g. `irb_05`).
- `hl_XX` — human-line public participants captured outside of IRB (e.g. `hl_02`).
- Survey JSON and de-identified transcript directories reuse these identifiers to keep datasets aligned across `transcripts/`, `transcripts_de_ided/`, and `surveys/`.

## Repo Structure

Entries labelled with `_not tracked_` below are ignored locally and not committed to git.

### Top-level files

READMES:

- [PAPER_TODO.md](PAPER_TODO.md) — notes and todos related to the paper text and figures.
- [README.md](README.md) — this top-level guide.
- [README_Parse.md](README_Parse.md) — detailed parsing and de-identification workflow.
- [README_Annotation.md](README_Annotation.md) — detailed annotation and classification workflow.
- [README_Subsets.md](README_Subsets.md) — detailed subset creation, scoring, and paraphrasing workflow.
- [README_Analyses.md](README_Analyses.md) — detailed analysis scripts and tables.
- [LICENSE](LICENSE) — project license.

Config files:

- [Makefile](Makefile) — automation entry points (`init`, `pyfmt`, `viewer`, etc.).
- [setup.py](setup.py) — legacy installer shim for packaging.
- [package.json](package.json) — JavaScript package definitions for the dashboards and viewer.
- [package-lock.json](package-lock.json) — npm lockfile pinning JavaScript dependencies.
- [pyproject.toml](pyproject.toml) — Python build metadata plus black/isort configuration.
- [requirements.txt](requirements.txt) — Python dependencies used by scripts and tooling.
- [uv.lock](uv.lock) — lockfile for `uv`-based Python dependency management.
- [eslint.config.mjs](eslint.config.mjs) — lint configuration for JavaScript and TypeScript code.
- [.gitattributes](.gitattributes) — Git attributes, including Git LFS settings.
- [.gitignore](.gitignore) — patterns for files and directories to ignore in Git.
- [.prettierignore](.prettierignore) — files and directories excluded from Prettier formatting.
- [.eslintcache](.eslintcache) _not tracked_ — cached eslint results (safe to delete).

Data files:

- [qualtrics.csv](qualtrics.csv) _not tracked_ — latest Qualtrics export fetched manually.

Directories:

- `manual_annotation_labels/` — human label outputs organized by annotator.
- _`env-delusions/`_ — local Python virtual environment created by `make init`.

### Code

#### src/

Main Python package code for the project.

- `analysis_utils/` — shared helpers and utilities used by analysis scripts.
- `annotation/` — code used by annotation scripts and viewers.
- `chat/` — chat helper utilities and IO functions.
- `chatlog_processing_pipeline/` — ingestion and normalization logic for raw transcripts.
- `data/` — static data assets such as `annotations.csv`.
- `llm_utils/` — shared tooling for calling LLMs and managing prompts.
- `utils/` — general-purpose utilities reused across modules.

#### scripts/

Helper utilities and entry points for transcript and annotation workflows.

- `annotation/` — LLM-driven annotation and agreement tooling.
- `annotation/classify_chats.py` — orchestrates LLM classification runs over transcripts.
- `annotation/summarize_annotation_positives.py` — aggregates positive counts across JSONL outputs from `classify_chats.py`.
- `annotation/annotation_conversation_counts.py` — per-annotation conversation-level summaries and CSV exports.
- `annotation/prepare_manual_annotation_dataset.py` — builds manual annotation datasets from model outputs.
- `annotation/compute_annotation_agreement.py` — computes agreement metrics across annotators.
- `annotation/export_conversation_tables.py` — exports per-message tables for analysis.

- `check_repo_sizes.sh` — simple shell helper to inspect large files and directories.

- `misc/` — miscellaneous utilities not tied to a single pipeline stage.
- `misc/qualtrics_to_json.py` — converts survey CSV exports into JSON files.
- `misc/compute_vader_sentiments.py` — optional sentiment scores over transcripts using VADER.
- `misc/generate_eval_instances.py` — builds evaluation instances from chats.
- `misc/generate_tabular_chats_from_annotations.py` — tabular exports from annotation outputs.

- `parse/` — parsing and reviewer export helpers.
- `parse/parse_plan.py` — drives plan-based parsing for tricky transcripts via `transcripts/plan.csv`.
- `parse/format_chats_html.py` — renders transcripts for reviewers as HTML suitable for Google Docs.
- `parse/compute_participant_ordering_and_stats.py` — computes per-participant transcript statistics and ordering categories.

- `subsets/` — subset selection, scoring, and paraphrasing utilities.
- `subsets/make_subsets.py` — extracts message windows around quoted anchors.
- `subsets/classify_subsets.py` — scores subsets with LLMs.
- `subsets/summarize_subset_harmfulness.py` — merges quality and harmfulness scores.
- `subsets/paraphrase_subsets.py` — paraphrases subsets while preserving metadata.
- `annotations.py` — default prompt templates and metadata shared across scripts.

#### analysis/

Python scripts, cached data, and figures for the main quantitative analyses.

Scripts:

- `agreement/` — helpers and assets for annotation agreement analyses.
- `latex/` — LaTeX helper files or fragments related to the paper.
- `viewer/` — small viewer app assets for browsing analysis outputs.
- `viewer/no_cache_http_server.py` — local HTTP server backing `make viewer` and annotation dashboards.

- `agreement_pr.py` — computes precision–recall–style metrics and plots for agreement runs.
- `compute_annotation_frequencies.py` — builds global and marginal annotation frequency tables.
- `compute_annotation_length_effects.py` — analyzes annotation behavior as a function of message length.
- `compute_annotation_topics.py` — prepares topic-modeling inputs from annotation matches.
- `compute_demographics.py` — summarizes participant demographics from survey and transcript metadata.
- `compute_participant_annotation_profiles.py` — builds participant-by-annotation profiles and clustered heatmaps.
- `compute_sequential_annotation_dynamics.py` — computes within-conversation sequential annotation transition matrices.
- `demo_join_annotations_transcripts.py` — example join of annotation and transcript tables.
- `make_participant_plots.py` — participant-level plotting helper script.
- `participant_ordering.json` — cached participant ordering metadata reused across analyses.
- `plot_annotation_length_effects.py` — plotting frontend for length-effect analysis results.
- `plot_annotations_by_ppts.py` — participant-level time and sequence plots from annotations.
- `preprocess_annotation_family.py` — normalizes raw annotation JSONL into compact Parquet tables.

Output files:

- `data/` — cached CSV and Parquet outputs from analysis scripts.
- `figures/` — generated figures (PDF/PNG) used in the paper and dashboards.

#### analysis_dashboards/

Interactive dashboards for exploring annotation and analysis outputs.

- `annotation_dashboard_noTimestamps.py` — dashboard for annotation summaries without timestamp-based views.
- `annotation_dashboard_timestamps.py` — dashboard variant including timestamp-aware views.
- `dashboard_common.py` — shared helpers and layout components used by the dashboards.

### Data

#### surveys/

cached survey JSON exports keyed by participant.

- `irb_02.json`–`irb_15.json` — per-participant survey responses for IRB participants (one file per participant).

#### transcripts/

TODO: ref to gdrive storage

_not tracked_ — working transcripts and metadata used as raw inputs.

- `01_original/` — raw transcript uploads as delivered from collection.
- `02_parsed/` — conversation-turn JSON grouped by participant after parsing.
- `metadata.csv` — survey metadata aligned with transcripts.
- `old_parsed/` — legacy parsed transcripts kept for reference.
- `plan.csv` — plan file driving `parse_plan.py` for complex inputs.

#### transcripts_data/

Columnar tables derived from de-identified transcripts.

- `transcripts_index.parquet` — lightweight index of all messages and metadata without content.
- `transcripts.parquet` — full message table including `content` for selected analyses.

#### transcripts_de_ided/

Redacted transcripts stored with Git LFS.

- `human_line/` — anonymized public transcripts (e.g. `hl_02/`).
- `public/` — anonymized non-IRB public transcripts.
- `under_irb/` — anonymized IRB sets such as `irb_05/` with JSON and HTML outputs.

#### _annotation_outputs/_

TODO: ref to Gcloud download above

Sample and working JSONL outputs produced by `classify_chats.py`.

- `human_line/` — annotation outputs for public human-line participants.
- `under_irb/` — annotation outputs for IRB-protected participants (e.g. `irb_05/` runs).

#### annotations/

Preprocessed annotation tables used by downstream scripts.

- `all_annotations.parquet` — consolidated annotations table across job families.
- `all_annotations__preprocessed.parquet` — preprocessed per-message annotation table.
- `all_annotations__matches.parquet` — compact table of messages with veridical quote matches.

#### manual_annotation_inputs/

- `manual_annotation_inputs/` — JSONL inputs prepared for manual annotation rounds.
- `20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl` — manual annotation input sample capped at 100 items with context.
- `20251210-083216__model=gpt-5.1&per-annotation=10.from-100.jsonl` — curated manual annotation input with 10 examples per annotation.
- `20251218-091458__model=gpt-5.1&max-items=10&preceding-context=0&llm-score-cutoff=5.jsonl` — small manual annotation input filtered by LLM score cutoff.
- `test.jsonl` — tiny test input file for manual annotation tooling.

#### subsets/

Focused subsets of transcripts plus scoring outputs.

- `annotation_conversation_counts/` — conversation-level CSVs from annotation count runs.
- `annotation_outputs/` — annotation outputs computed over subset transcripts.
- `auto_subsets/` — auto-generated subset plans derived from annotation statistics.
- `items/` — subset items generated from particular annotation plans.
- `subset_quality.jsonl` — quality scores for subsets from a particular run.
- `subset_quality_scores.csv` _not tracked_ — spreadsheet-aligned summary of subset quality and harmfulness flags.
- `subsets.csv` — canonical manual subsets plan aligned with the paper.
