# Analyses

TODO: summarize

## Contents

- [Preprocessing](#preprocessing)
- [Exporting conversations to tables](#exporting-conversations-to-tables)
- [Transcript tables for analysis](#transcript-tables-for-analysis)
- [Participant demographics from surveys](#participant-demographics-from-surveys)
- [Participant ordering and transcript stats](#participant-ordering-and-transcript-stats)
- [Global annotation frequency table](#global-annotation-frequency-table)
- [Per-participant annotation trajectories (time and sequence)](#per-participant-annotation-trajectories-time-and-sequence)
- [Per-participant stacked annotation streamgraphs](#per-participant-stacked-annotation-streamgraphs)
- [Conversation counts (currently unused)](#conversation-counts-currently-unused)
- [Annotation profiles](#annotation-profiles)
- [Sequential dynamics](#sequential-dynamics)
- [Topic-modeling inputs from annotation matches](#topic-modeling-inputs-from-annotation-matches)
- [Optional topic modeling analyses](#optional-topic-modeling-analyses)

## Preprocessing

### Annotations

```bash
python analysis/preprocess_annotation_family.py \
  annotation_outputs/under_irb/irb_01/all_annotations__part-0001.jsonl \
  --annotations-csv src/data/annotations.csv \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json \
  --output annotations/all_annotations__preprocessed.parquet
```

TODO: why do this, what does it do

## Exporting conversations to tables

To export per-message CSV tables for analysis (legacy path; prefer the Parquet export below for new work), use:

```
python scripts/annotation/export_conversation_tables.py
```

By default this writes one CSV per transcript under `conversation_tables/`, mirroring the layout of `transcripts_de_ided/` (for example `transcripts_de_ided/under_irb/irb_05/chat.html.json` becomes `conversation_tables/under_irb/irb_05/chat.html.json.csv`). Each row contains participant id, conversation key and index, message index, role, content, and timestamps. To add annotation presence columns (one per annotation id, thresholded by `--min-score`) and links back to the JSONL annotation files, re-run with:

```
python scripts/annotation/export_conversation_tables.py --include-annotations
```

- You can edit an annotation’s name/description directly in the UI, then run a guarded "Regenerate" flow that performs a dry run first and surfaces an estimated max cost before proceeding. The dialog exposes key parameters and supports re-running with the exact set of annotations used by the currently loaded dataset.
- When multiple classification runs exist for the same annotation, the viewer lets you select which run to display (newest by default), and it shows the snapshot of prompt inputs (name/description, system prompt, template) used for that run.

### Transcript tables for analysis

To build columnar tables for all transcript messages (separate from annotations), run:

```bash
python scripts/parse/export_transcripts_parquet.py \
  --transcripts-root transcripts_de_ided \
  --output-dir transcripts_data
```

This writes:

- `transcripts_data/transcripts_index.parquet` – one row per message with keys and lightweight metadata:
  - `participant`, `source_path`, `chat_index`, `message_index`, `role`, `timestamp`, `chat_key`, `chat_date`.
- `transcripts_data/transcripts.parquet` – the same rows plus a `content` column.

These tables are designed to be joined with annotation tables on the shared location key (`participant`, `source_path`, `chat_index`, `message_index`) without repeatedly reparsing the raw JSON under `transcripts_de_ided/`.

#### Example: join annotations to transcript index (no content)

With the preprocessed annotations and transcript index in place, you can do a fast, content-free join in a notebook or script:

```python
import pandas as pd
from pathlib import Path

annotations_path = Path("annotations/all_annotations__preprocessed.parquet")
index_path = Path("transcripts_data/transcripts_index.parquet")

annotations = pd.read_parquet(annotations_path)
transcript_index = pd.read_parquet(index_path)

merged = annotations.merge(
    transcript_index,
    on=["participant", "source_path", "chat_index", "message_index"],
    how="inner",
    suffixes=("_ann", "_tx"),
)
```

`merged` now contains per-message annotation scores plus transcript-level metadata (role, timestamp, conversation key/date) without loading any `content`.

#### Example: attach content for a specific message

When you need the full text for a small subset of messages, you can look up `content` on demand from `transcripts.parquet`:

```python
from pathlib import Path

transcripts_path = Path("transcripts_data/transcripts.parquet")

# Pick one joined row as an example.
row = merged.iloc[0]

filters = [
    ("participant", "=", row["participant"]),
    ("source_path", "=", row["source_path"]),
    ("chat_index", "=", int(row["chat_index"])),
    ("message_index", "=", int(row["message_index"])),
]

transcript_row = pd.read_parquet(
    transcripts_path,
    engine="pyarrow",
    filters=filters,
)

print(transcript_row["content"].iloc[0])
```

This pattern keeps most analysis on the lightweight index and annotations tables, only touching `content` for the specific messages you need.

<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Participant demographics from surveys

To compute basic participant demographics (age and gender) from the IRB survey JSON files in `surveys/` and write a CSV suitable for paper tables, run:

```bash
python analysis/compute_demographics.py \
  --surveys-dir surveys \
  --output analysis/data/demographics.csv
```

This script:

- Scans `irb_*.json` files in the specified surveys directory (by default `surveys/`), using the keys `"What is your age?"` and `"What is your gender? - Selected Choice"`.
- Prints a text summary of age range, mean, median, and gender breakdown to the console.
- Writes a long-format CSV to `analysis/data/demographics.csv` with fields, categories, counts, and percentages that can be dropped directly into a LaTeX table.

## Participant ordering and transcript stats

To compute per-participant ordering categories (for downstream sequential analyses) and high-level transcript statistics that do not depend on annotation outputs, run:

```bash
python scripts/parse/compute_participant_ordering_and_stats.py \
 --transcripts-root transcripts_de_ided \
 --ordering-json analysis/participant_ordering.json \
 --stats-csv analysis/data/participant_transcript_stats.csv
```

This scans `transcripts_de_ided` using the same linearized, visible user/assistant message paths as the rest of the pipeline and writes:

- `analysis/participant_ordering.json`: ordering category per participant (for example, whether a global ordering over messages is available).
- `analysis/data/participant_transcript_stats.csv`: per-participant summary table (conversation and message counts, lengths, file types, and model usage) sorted by total messages with a final `TOTAL` row.

### By participant plots

We currently have static analyses available to run on the de-ided transcripts. These produces various graphs in `analysis/figures` for each of the participants passed (look at the subdirectories).

Example analysis artifacts:

- [conversation_counts.png](analysis/figures/conversation_counts.png) summarises the total chats per participant.
- [irb_05 sequence summary](analysis/figures/participants/irb_05/irb_05_message_level_summary.png) shows message mix for a single participant.

<img src="analysis/figures/conversation_counts.png" alt="Conversation counts bar chart" width="320" />

```bash
python analysis/make_participant_plots.py transcripts_de_ided/
```

To run just the analyses for one participant:

```bash
python analysis/make_participant_plots.py transcripts_de_ided/ --participants irb_05
```

## Global annotation frequency table

To compute the global and marginal annotation frequencies used in the paper, run:

```bash
python analysis/compute_annotation_frequencies.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json \
    --annotations-csv src/data/annotations.csv \
    --long-quantile 0.75 \
    --output analysis/data/annotation_frequencies.csv
```

This command:

- First materialises a per-message Parquet table (`annotations/all_annotations__preprocessed.parquet`) that aggregates scores across the full job family discovered from the reference JSONL.
- Then applies per-annotation score cutoffs from the metrics JSON file to compute message-pooled, participant-averaged, and length-enrichment statistics for each annotation.
- Writes the canonical frequency table to `analysis/data/annotation_frequencies.csv`.

## Analyses on Conversation Length

### Per-message regression

A per-message regression where the outcome is the (log) total length of the conversation that message belongs to, and the predictor is whether that message is annotated (optionally using cluster-robust SEs by participant).

```bash
python analysis/compute_annotation_length_effects.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json \
    --output analysis/data/annotation_length_effects.csv \
    --cluster-by-participant

# Plots all of the annotations
python analysis/plot_annotation_length_effects.py \
    --input analysis/data/annotation_length_effects__logistic.csv \
    --output analysis/figures/annotation_length_effects.pdf

# Alternatively just plot some of the annotations
python analysis/plot_annotation_length_effects.py \
        --input analysis/data/annotation_length_effects__logistic.csv \
        --output analysis/figures/annotation_length_effects_extremes.pdf \
        --max-bottom 5 \
        --max-top 5
```

[analysis/figures/annotation_length_effects.pdf](analysis/figures/annotation_length_effects.pdf)

### GLMM

A conversation-level GLMM where the outcome is the total number of messages in a conversation (on the log Poisson scale), and the predictors summarize that annotation's dynamics within the conversation: how many positive messages it has, and when the first and last positives occur. These coefficients tell us how much longer or shorter conversations tend to be, on average. We use per-participant effects.

I find no real effect modeling with the first or last positive (`--predictor last_pos_index` or `--predictor first_pos_index`) so I'm not sure what this gives us more than the original regression above. Mabe just statistical validity on the individual participant's effects?

```bash
Rscript analysis/conversation_length_glmm.R

python analysis/plot_conversation_length_glmm_effects.py \
    --model-type glm_participant_fixed \
    --predictor n_pos
```

[analysis/figures/conversation_length_glmm_effects.pdf](analysis/figures/conversation_length_glmm_effects.pdf)

## Per-participant annotation trajectories (time and sequence)

To generate static per-participant annotation trajectories (both time-based and sequence-based) plus overall per-annotation summaries from a full job family, use:

```bash
python analysis/plot_annotations_by_ppts.py \
  annotation_outputs/under_irb/irb_01/all_annotations.jsonl \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json \
  --participant-ordering-json analysis/participant_ordering.json \
  --output analysis/figures
```

This command:

- Discovers the full job family sharing the basename of `all_annotations.jsonl` under `annotation_outputs/`.
- Applies per-annotation LLM score cutoffs from the metrics JSON (or a global `--score-cutoff` when no JSON is provided).
- Uses participant ordering metadata from `analysis/participant_ordering.json` to decide when time-based plots are valid.
- Writes:
  - Per-participant figures under `analysis/figures/participants/<ppt_id>/annotations/`:
    - Time-based rolling trajectories over 5-day windows.
    - Sequence-based rolling trajectories over 20-message windows (with dual y-axes for proportions and counts).
  - Overall per-annotation summary figures under `analysis/figures/annotations_overall/`, including 95% confidence intervals over participants.

How to run multiple participants:

```
python scripts/annotation/classify_chats.py --input transcripts_de_ided \
--participant hl_01 \
--participant hl_06 \
--participant hl_07 \
--participant hl_08 \
--participant hl_12 \

TODO: make sure this is up to date
--participant irb_03 \
--participant irb_05 \
--participant irb_06 \
--annotation assistant-grand-significance-ideas \
--dry-run
```

`gpt-5.1` is the default model but you can supply any OpenAI, Anthropic, TogetherAI, etc. model as interpretable by LiteLLM (so long as you have the credentials). Do not use reasoning-style models (for example, `gpt-5`) for these classification runs.

## Per-participant stacked annotation streamgraphs

To visualize how multiple annotations co-occur over the course of each participant's interaction and in aggregate, use stacked streamgraphs:

```bash
python analysis/plot_annotation_streamgraphs_by_ppts.py \
  annotation_outputs/under_irb/irb_01/all_annotations.jsonl \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json \
  --participant-ordering-json analysis/participant_ordering.json \
  --output analysis/figures
```

This command writes:

- Per-participant stacked-prevalence PDFs under `analysis/figures/participants/<ppt_id>/annotations/`, showing how the mix of annotation types evolves over normalized message index and (when dates are available) over time.
- Overall stacked-prevalence PDFs under `analysis/figures/annotations_overall/`, summarizing how annotation combinations evolve across participants.

These plots complement the time-window trajectory charts by emphasizing relative composition and co-occurrence of annotations rather than individual code trajectories.

## Conversation counts (currently unused)

To aggregate at the conversation level for a single annotation id (for example, to see how many conversations have at least `N` positive messages for that code), use:

```bash
python scripts/annotation/annotation_conversation_counts.py \
  annotation_outputs/human_line/hl_01/20251222-192416__input=transcripts_de_ided\&max_messages=1000\&model=gpt-5.1\&preceding_context=3\&randomize=True\&randomize_per_ppt=equal.jsonl \
  --annotation-id grand-significance \
  --score-cutoff 5 \
  --min-occurrences 2
```

This scans all sibling JSONL files with the same basename under the outputs root, computes per-conversation positive and total message counts for the requested annotation, filters to conversations with at least `--min-occurrences` positives (respecting the score cutoff), and writes a CSV under `analysis/data/annotation_conversation_counts/` whose filename encodes the annotation id, cutoff, and minimum-occurrence parameters.

## Annotation profiles

To compute participant-level annotation profiles and clustered heatmaps from a full job family, use:

```bash
python analysis/compute_participant_annotation_profiles.py \
  annotation_outputs/under_irb/irb_01/all_annotations.jsonl \
  --onset-threshold-k 5 \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json \
  --output analysis/data/participant_annotation_profiles.csv
```

This writes an annotations-by-participants CSV table to `analysis/data/participant_annotation_profiles.csv` and saves a clustered heatmap and dendrogram as PDFs under `analysis/figures/`.

## Sequential dynamics

To compute within-conversation sequential annotation dynamics (for K = 0, 1, and 10) from a full job family, first generate the per-K CSV tables:

```bash
python analysis/compute_sequential_annotation_dynamics.py \
  annotation_outputs/under_irb/irb_01/all_annotations.jsonl \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json \
  --output-prefix analysis/data/sequential_dynamics
```

This writes per-K X->Y matrix and top-pairs CSVs under `analysis/data/` (for example, `analysis/data/sequential_dynamics_K10_matrix.csv` and `analysis/data/sequential_dynamics_K10_top_pairs.csv`).

When you instead use a global LLM score cutoff via `--llm-score-cutoff N` (and omit `--llm-cutoffs-json`), the same files are written with an explicit cutoff suffix in the prefix. For example:

```bash
python analysis/compute_sequential_annotation_dynamics.py \
  annotations/all_annotations__preprocessed.parquet \
  --llm-score-cutoff 10 \
  --window-k 0 --window-k 1 --window-k 2 --window-k 5 --window-k 10 --window-k 100
```

will write files such as `analysis/data/sequential_dynamics__scorecutoff10_K10_matrix.csv`.

To render a combined log-enrichment heatmap PDF for the same set of K values, run:

```bash
python analysis/plot_sequential_annotation_dynamics.py \
  --output-prefix analysis/data/sequential_dynamics \
  --window-k 0 --window-k 1 --window-k 10 \
  --figure-path analysis/figures/sequential_enrichment_Ks.pdf
```

This writes a single combined log-enrichment heatmap PDF for all requested K values to `analysis/figures/sequential_enrichment_Ks.pdf`.

If you used a global score cutoff, pass the suffixed prefix instead, for example:

```bash
python analysis/plot_sequential_annotation_dynamics.py \
  --output-prefix analysis/data/sequential_dynamics__scorecutoff10 \
  --window-k 0 --window-k 1 --window-k 2 --window-k 5 --window-k 10 --window-k 100 \
  --figure-path analysis/figures/sequential_enrichment_Ks__scorecutoff10.pdf
```

which uses the `sequential_dynamics__scorecutoff10_K*_matrix.csv` files as inputs for the heatmaps.

## Topic-modeling inputs from annotation matches

To prepare a corpus of messages with veridical quote matches for topic modeling, use:

```bash
python analysis/compute_annotation_topics.py \
  annotations/all_annotations__matches.parquet \
  --output analysis/data/annotation_topics_all_annotations.jsonl
```

This command:

- Uses `preprocess_annotation_family.py` to:
  - Discover the job family that shares the basename of the reference JSONL under `annotation_outputs`.
  - Load per-annotation LLM score cutoffs from the metrics JSON.
  - Select only records where every quoted span in `matches` is an exact substring of the original `content`, ensuring that all included examples have veridical matches.
  - Write a directory of compact matches CSV shards under `annotations/all_annotations__matches`, each capped at roughly 50MB, containing one row per validated match record, and a consolidated matches Parquet file at `annotations/all_annotations__matches.parquet`.
- Then converts these matches records into a JSONL file with one JSON object per selected message; downstream tools can load this file into a topic-modeling library of choice.

## Optional topic modeling analyses

These topic modeling commands are optional for the main paper and are intended for exploratory analyses.

To compute global topics and topic distributions for annotations and participants, run:

```bash
python analysis/compute_annotation_topics.py \
    --annotations-parquet annotations/all_annotations__preprocessed.parquet \
    --transcripts-parquet transcripts_data/transcripts.parquet
```

Key outputs under `analysis/data/annotation_topics_artifacts/`:

- `global_topics.json` and `global_topics_meta.json`: global topic definitions, sizes, and terms.
- `message_topics.csv`: per-message topic assignments.
- `annotation_summary.csv` and `annotation_topic_enrichment.csv`: how each annotation is distributed over topics and which topics are most enriched.
- `participant_topics.csv`: topic distributions and enrichment scores per participant.

To turn these artifacts into figures and topic-term plots, run:

```bash
python analysis/plot_annotation_topics.py
```

- **Outputs (figures):**
  - `analysis/figures/annotation_topic_heatmap.pdf`: annotation-by-topic enrichment heatmap.
  - `analysis/figures/annotation_entropy_bar.pdf`: per-annotation topic heterogeneity bar chart.
  - `analysis/figures/participant_topic_heatmap.pdf`: participant-by-topic enrichment heatmap.
  - `analysis/figures/topics/topic_XXX_terms.pdf`: one topic-term bar chart PDF per topic.

For the paper, these outputs can support:

- High-level tables summarizing how narrowly or broadly each annotation is distributed over topics.
- Example topic-term bar charts to illustrate the main themes behind key annotations.
- Heatmaps that show which topics are most associated with specific annotations or participants, ideally placed in the appendix.
