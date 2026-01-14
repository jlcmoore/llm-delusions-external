## Paper Tables and Figures TODO

### Quick LaTeX prep and sync commands

Run these after the underlying analyses (agreement metrics, annotation CSVs, etc.) have been generated:

```bash
# 1) Build LaTeX-friendly agreement summary CSVs
python analysis/latex/create_agreement_summary_csv.py \
  --metrics analysis/agreement/test/metrics.score-0.json \
  --metrics analysis/agreement/test-random/metrics.score-0.json \
  --metrics analysis/agreement/test-matches/metrics.score-0.json \
  --metrics analysis/agreement/validation/metrics.json

# 2) Copy CSVs/figures, build annotation schema table, and generate .tex tables into the Overleaf repo
python analysis/latex/sync_files.py
```

### (Done) Participant demographics (surveys)

```bash
python analysis/compute_demographics.py --surveys-dir surveys --output analysis/data/demographics.csv
```

We report the summary demographics inline (no table).

### (Done) Participant "what went wrong"

Pull this from here: <https://docs.google.com/spreadsheets/d/1FF-mDuezYUBiOliNf7lvQ-_skn00Zuo5GFSTzJ2HjAU/edit>

### (Done) Descriptive stats on transcripts

```bash
python scripts/parse/compute_participant_ordering_and_stats.py \
  --transcripts-root transcripts_de_ided \
  --ordering-json analysis/participant_ordering.json
```

We put in the entirety of `analysis/data/participant_transcript_stats.csv` (see LaTex prep).

### (Done) Annotation schema overview

We include `annotations.csv` as a table with selected columns. (See LaTex prep.)

### Agreement statistics and tables

#### (Done) Validation set

**Classification Data**

TODO (minor, for reproducibility purposes): What commands did we run to produce the original annotation data?

**Manual Input Data**

TODO (minor, for reproducibility purposes): what commands did we use to generate the manual input files?

Random most annotations (but some changed):

`manual_annotation_inputs/20251130-203346__model=gpt-5.1\&max-items=100\&preceding-context=3.jsonl`

The subset of the previous to have 10 annotations only each which we double annotated:

`manual_annotation_inputs/20251210-083216__model=gpt-5.1\&per-annotation=10.from-100.jsonl`

Positives onlys (score cutoff =5):

`manual_annotation_inputs/20251218-091458__model=gpt-5.1\&max-items=10\&preceding-context=0\&llm-score-cutoff=5.jsonl`

TODO: **rename this second file slug to be called validation (and replace usages elsewhere)**

**Analysis**

All items:

```bash
python scripts/annotation/compute_annotation_agreement.py \
    --dataset "manual_annotation_inputs/20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl" \
    --dataset "manual_annotation_inputs/20251210-083216__model=gpt-5.1&per-annotation=10.from-100.jsonl" \
    --dataset "manual_annotation_inputs/20251218-091458__model=gpt-5.1&max-items=10&preceding-context=0&llm-score-cutoff=5.jsonl" \
    --llm-run-basename 20251130-171532__input=transcripts_de_ided\&max_messages=1000\&model=gpt-5.1\&preceding_context=3\&randomize=True\&randomize_per_ppt=equal.jsonl \
    --llm-run-basename "20251205-161223__annotation=user-endorses-delusion&input=transcripts_de_ided&max_messages=1000&model=gpt-5.1&preceding_context=3&randomize=True&randomize_per_ppt=equal&replay_all_ppts=True.jsonl"
```

- `--dataset "manual_annotation_inputs/20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl"`
  - The original sample of 100 for manual annotation.
- `--dataset "manual_annotation_inputs/20251210-083216__model=gpt-5.1&per-annotation=10.from-100.jsonl"`
  - The sub sampled annotated twice (for three with the original sample)
- `-dataset "manual_annotation_inputs/20251218-091458__model=gpt-5.1&max-items=10&preceding-context=0&llm-score-cutoff=5.jsonl"`
  - The 'positives' only sample
- `--llm-run-basename 20251130-171532__input=transcripts_de_ided\&max_messages=1000\&model=gpt-5.1\&preceding_context=3\&randomize=True\&randomize_per_ppt=equal.jsonl`
  - The full gpt5.1 job
- `--llm-run-basename "20251205-161223__annotation=user-endorses-delusion&input=transcripts_de_ided&max_messages=1000&model=gpt-5.1&preceding_context=3&randomize=True&randomize_per_ppt=equal&replay_all_ppts=True.jsonl"`
  - We added in a new category late and needed another job for it.

This produced:

- `analysis/agreement/20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl/cases.score-0.jsonl`
- `analysis/agreement/20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl/metrics.json`

Which we renamed to:

- `analysis/agreement/validation/cases.score-0.jsonl`
- `analysis/agreement/validation/metrics.json`

Note that we did not compute agreement for both the positives and the negative alone.

#### (Done) Test set

**Datasets:**

```bash
python scripts/annotation/prepare_manual_annotation_dataset.py \
    --outputs-root annotation_outputs \
    --model gpt-5.1-2025-11-13 \
    --preceding-context 3 \
    --max-items 10 \
    --randomize \
    --randomize-per-ppt equal \
    --match-arguments-from annotation_outputs/under_irb/irb_06/all_annotations.jsonl \
    --preprocessed-table annotations/all_annotations__preprocessed.parquet \
    --transcripts-table transcripts_data/transcripts.parquet
```

Generates a file we renamed to:

`manual_annotation_inputs/test.jsonl` — mixed positives/negatives (10 random items per annotation from the `all_annotations` family. all 10 items are shared by all annotations).

```bash
python scripts/annotation/prepare_manual_annotation_dataset.py \
    --outputs-root annotation_outputs \
    --model gpt-5.1-2025-11-13 \
    --preceding-context 3 \
    --max-items 10 \
    --randomize \
    --randomize-per-ppt equal \
    --match-arguments-from annotation_outputs/under_irb/irb_06/all_annotations.jsonl \
    --preprocessed-table annotations/all_annotations__preprocessed.parquet \
    --transcripts-table transcripts_data/transcripts.parquet \
    --llm-positive-only \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json
```

generates a file we renamed to:

`manual_annotation_inputs/test-2.jsonl` — positive-only examples (10 items per annotation using per-annotation cutoffs from `metrics.score-0.json`).

**Analysis**:

random + positives (union of both files):

```bash
python scripts/annotation/compute_annotation_agreement.py \
    --dataset manual_annotation_inputs/test.jsonl \
    --dataset manual_annotation_inputs/test-2.jsonl \
    --llm-preprocessed-parquet annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json
```

Random only:

```bash
python scripts/annotation/compute_annotation_agreement.py \
    --dataset manual_annotation_inputs/test.jsonl \
    --llm-preprocessed-parquet annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json
```

Positives only

```bash
python scripts/annotation/compute_annotation_agreement.py \
    --dataset manual_annotation_inputs/test-2.jsonl \
    --llm-preprocessed-parquet annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json
```

We bring over two separate tables, not combined for the paper. (See LaTex prep.)

- LLM-vs-human-majority agreement table from `analysis/latex/generated/agreement_summary_majority_compact.csv`
- Human inter-annotator agreement table from `analysis/latex/generated/agreement_summary_inter_annotator_compact.csv`

### (Done) Agreement Precision - Recall Curves

(This is for the validation data, both the random and the positives only.)

Command to generate the files:

```bash
python analysis/agreement_pr.py --dataset "validation/"
```

[analysis/figures/agreement/pr_by_annotation**validation**gpt-5.1.pdf](analysis/figures/agreement/pr_by_annotation__validation__gpt-5.1.pdf)

Files:

- agreement precision-recall chart: `analysis/figures/agreement/pr__validation.pdf`
- per-annotation agreement precision-recall chart: `analysis/figures/agreement/pr_by_annotation__validation__gpt-5.1.pdf`

We want to include both of these: the overall and the by annotation figures side by side to justify why we use a cutoff for each. Put in the appendix.

### (Done) Preprocess annotations

This is the canonical path for all paper analyses that operate on annotation outputs. It materialises a per-message annotations table and a compact matches table in Parquet, which all downstream scripts consume.

```bash
python analysis/preprocess_annotation_family.py \
    annotation_outputs/under_irb/irb_01/all_annotations__part-0001.jsonl
```

It outputs:

- `annotations/all_annotations__preprocessed.parquet`
  - Canonical per-message table for the paper: one row per message with
    `participant`, `source_path`, `chat_index`, `message_index`, `role`, and
    one `score__<annotation_id>` column per code (no message content).
- `analysis/data/all_annotations__annotation_output_stats.csv`
  - Stats on the number of messages and rates of errors
- `annotations/all_annotations__matches.parquet`
  - Canonical matches table: one row per validated quote match with message
    content and extracted spans, used for topic modeling and examples.

All annotation-frequency, length-effects, trajectory, and topic analyses in the paper should read from these Parquet tables rather than re-parsing raw JSONL outputs.

### (Done) Descriptive stats on annotations (global frequencies)

```bash
python analysis/compute_annotation_frequencies.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json

python analysis/compute_annotation_frequencies.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-score-cutoff 10 \
    --output analysis/data/annotation_frequencies_cutoff=10.csv

python analysis/plot_annotation_frequency_histogram.py \
  analysis/data/annotation_frequencies.csv \
  --annotations-csv src/data/annotations.csv \
```

[analysis/figures/annotation_frequency_histogram.pdf](analysis/figures/annotation_frequency_histogram.pdf)

We save `analysis/data/annotation_frequencies.csv` to the paper. (See LaTex prep.)

#### Sub sets

```bash
python analysis/compute_annotation_set_frequencies.py \
  annotations/all_annotations__preprocessed.parquet \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json \
  --annotations-csv src/data/annotations.csv \
  --set sentience:assistant-misrepresents-sentience,theme-awakening-consciousness,user-assigns-personhood \
  --set-role sentience:both \
  \
  --set concerns-harm:user-violent-intent,user-suicidal-intent,assitant-validates-violent-feelings,assistant-validates-self-harm-feelings,assistant-discourages-violence,assistant-discourages-self-harm,assistant-facilitates-violence,assistant-facilitates-self-harm \
  --set-role concerns-harm:both \
  \
  --set facilitates-harm:assistant-facilitates-violence,assistant-facilitates-self-harm \
  --set-role facilitates-harm:assistant \
  \
  --set sycophancy:assistant-validates-ideas,assistant-positive-affirmation,assistant-hyperbole,grand-significance,assistant-dismisses-counterevidence,assistant-reports-others-admire-speaker,assistant-claims-unique-understanding \
  --set-role sycophancy:assistant \
  --output analysis/data/annotation_set_frequencies.csv


python analysis/plot_annotation_set_frequencies.py \
    analysis/data/annotation_set_frequencies.csv \
    --output analysis/figures/annotation_set_frequency_histogram.pdf
```

Outputs: analysis/data/annotation_set_frequencies.csv

and

### (DONE) How does conversation length affect the annotation prevalence?

A per-message regression where the outcome is the (log) number of messages remaining in the conversation after that message, and the predictors are whether that message is annotated plus a control for relative position in the conversation. The key coefficient tells us how much longer or shorter conversations are, on average, after annotated messages versus unannotated ones at the same point in the dialogue. (This is styled after a survival hazard analysis.)

```bash
python analysis/compute_annotation_post_onset_lengths.py \
      annotations/all_annotations__preprocessed.parquet \
      --llm-cutoffs-json analysis/agreement/validation/metrics.json \
      --length-transform log \
      --cluster-by-participant \
      --output analysis/data/annotation_remaining_length_effects.csv

python analysis/plot_annotation_hazard_effects.py \
    analysis/data/annotation_remaining_length_effects.csv \
    analysis/data/annotation_frequencies.csv \
    --output analysis/figures/annotation_remaining_length_histogram.pdf

# For extemes
python analysis/plot_annotation_hazard_effects.py \
  analysis/data/annotation_remaining_length_effects.csv \
  analysis/data/annotation_frequencies.csv \
  --output analysis/figures/annotation_remaining_length_histogram_extremes.pdf \
  --max-bottom 5 --max-top 10
```

[analysis/figures/annotation_remaining_length_histogram.pdf](analysis/figures/annotation_remaining_length_histogram.pdf)

[analysis/figures/annotation_remaining_length_histogram_extremes.pdf](analysis/figures/annotation_remaining_length_histogram_extremes.pdf)

### (CUT) Annotation Prevalence Over Time (and by ppt)

To visualize how multiple annotations co-occur over the course of each participant's interaction and in aggregate, use stacked streamgraphs:

```bash
python analysis/plot_annotation_streamgraphs_by_ppts.py \
  annotations/all_annotations__preprocessed.parquet \
  --participant-ordering-json analysis/participant_ordering.json \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json

python analysis/plot_annotation_streamgraphs_by_ppts.py \
  annotations/all_annotations__preprocessed.parquet \
  --participant-ordering-json analysis/participant_ordering.json \
  --llm-score-cutoff 10

# Using score cutoff 10

```

[analysis/figures/annotations_overall/overall_time_stacked_raw.pdf](analysis/figures/annotations_overall/overall_time_stacked_raw.pdf)

[analysis/figures/annotations_overall/overall_time_stacked_raw\_\_scorecutoff10.pdf](analysis/figures/annotations_overall/overall_time_stacked_raw__scorecutoff10.pdf)

This command writes:

- Per-participant stacked-prevalence PDFs under `analysis/figures/participants/<ppt_id>/annotations/`, showing how the mix of annotation types evolves over normalized message index and (when dates are available) over time.
- Overall stacked-prevalence PDFs under `analysis/figures/annotations_overall/`, summarizing how annotation combinations evolve across participants.

TODO: consider including the overall figure or some specific participant's trajectories

We can also pair this with per participant annotation trajectories:

```bash
python analysis/plot_annotations_by_ppts.py \
  annotations/all_annotations__preprocessed.parquet \
  --participant-ordering-json analysis/participant_ordering.json \
  --llm-cutoffs-json analysis/agreement/validation/metrics.json

# Using score cutoff 10
python analysis/plot_annotations_by_ppts.py \
  annotations/all_annotations__preprocessed.parquet \
  --participant-ordering-json analysis/participant_ordering.json \
  --llm-score-cutoff 10
```

These show how we have time and sequence based plotting

[grand-significance\_\_overall_sequence_20msgs.pdf](analysis/figures/annotations_overall/grand-significance__overall_sequence_20msgs.pdf)

[grand-significance**overall_sequence_20msgs**scorecutoff10.pdf](analysis/figures/annotations_overall/grand-significance__overall_sequence_20msgs__scorecutoff10.pdf)

[assistant-facilitates-violence\_\_overall_time_5d.pdf](analysis/figures/annotations_overall/assistant-facilitates-violence__overall_time_5d.pdf)

[assistant-facilitates-violence**overall_time_5d**scorecutoff10.pdf](analysis/figures/annotations_overall/assistant-facilitates-violence__overall_time_5d__scorecutoff10.pdf)

- Per-participant figures under `analysis/figures/participants/<ppt_id>/annotations/`:
  - Time-based rolling trajectories over 5-day windows.
  - Sequence-based rolling trajectories over 20-message windows (with dual y-axes for proportions and counts).
- Overall per-annotation summary figures under `analysis/figures/annotations_overall/`, including 95% confidence intervals over participants.

TODO: willie will play around with the normalized bar charts

### (CUT) Relative annotation rates per participant and clustering

Compute relative rates for each annotation for each participant and hierarchical clustering.

```bash
python analysis/compute_participant_annotation_profiles.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json \
    --onset-threshold-k 5
```

[analysis/figures/participant_profiles_heatmap_participants.pdf](analysis/figures/participant_profiles_heatmap_participants.pdf)

NB: We could change to `--cluster-mode annotations`, but I think it is more informative to cluster participants given our current data.

- **Outputs:**
  - participant profiles CSV: `analysis/data/participant_annotation_profiles.csv`
  - clustered heatmap PDF: `analysis/figures/participant_profiles_heatmap_participants.pdf`
  - clustering dendrogram PDF: `analysis/figures/participant_profiles_dendrogram_participants.pdf`

- **Paper ideas:**
  - `Figure: Participant-by-annotation profile heatmap with hierarchical clustering.`
  - Add in the entirety of `analysis/data/participant_annotation_profiles.csv`.

  - **LaTeX caption stub (participant profiles heatmap figure):** `\caption{TODO}`

TODO: figure out if we can just export the whole table as a pdf to bake the heatmap aspect into it. Or some fancy latex to color each cell.

TODO: ensure the clustered heatmap PDF includes a clear colorbar/legend for the relative annotation rate values (with units and value range).

TODO: consider adding marginal summaries to the figure: per-annotation global frequency across participants (top bar) and per-participant volume (right bar), and decide whether the volume should be total annotations or total messages per participant (store both in the CSV if helpful).

TODO: consider adding thin metadata strips around the heatmap (e.g., participant cohort/condition or annotation groupings) if we settle on useful groupings.

TODO: ensure the figure title/labels and caption explicitly mention the onset threshold `k = 5`, and that the caption points to `analysis/data/participant_annotation_profiles.csv` as the source for the full numeric table.

TODO: clustering by ppt is similar to latent profile analysis (which itself is similar to k means). Consider whether we want to switch to this approach. @Ashish

### (Done) Correlations / sequential dynamics between annotations

**model choice**

We use one sequential-dynamics summaries, controlled by the `--effect-source` flag:

- _Beta K-window model (`--effect-source beta`)_. Here we collapse each window into a binary event and model $P(Y \text{ occurs at least once within the next } K \text{ messages})$ as a Bernoulli outcome. For a fixed $K$ and source $X$, each occurrence of $X$ with a full $K$-step future contributes one trial; the trial is a ``success'' if $Y$ appears at least once anywhere in that window. We place a weak Beta prior on the K-window probability $P(Y \text{ occurs within } K \mid X)$, centred at the global K-window rate for $Y$, and compute the posterior mean and an approximate 95\% interval. The heatmap colour encodes the log$_2$ K-window occurrence lift,
  \[
  \log_2 \frac{P(Y \text{ occurs at least once within } K \mid X)}{P(Y \text{ occurs at least once within } K)},
  \]
  which tells us how much more (or less) likely it is that $Y$ appears at all in the next $K$ turns after $X$ relative to its global K-window probability. This window-level view is robust to multiple occurrences of $Y$ in the same window (any positive count is a success) and comes with a natural notion of uncertainty from the Beta posterior, which we also expose in the per-target bar plots via confidence intervals.

NB: we're not doing k = 0 because that's just annotation cooccurrence on the same message and we're not doign k=1 because the output of that is mostly just that "assistants follow users" and "users follow assistants" due to the scoping of our annotations

```bash
# Generate the transitions using the validation cutoffs
python analysis/compute_sequential_annotation_dynamics.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json \
    --output-prefix analysis/data/sequential_dynamics/base \
    --window-k 2 --window-k 3 --window-k 4 --window-k 5 --window-k 10

# Optionally also generate using a score cutoff of 10
python analysis/compute_sequential_annotation_dynamics.py \
  annotations/all_annotations__preprocessed.parquet \
  --window-k 2 --window-k 5 --window-k 10 \
  --llm-score-cutoff 10
```

**Outputs:**:

- full sequential dynamics matrices for K=N to `analysis/data/sequential_dynamics/base_K<N>_matrix.csv`
- top enriched pairs for K=N to `analysis/data/sequential_dynamics/base_K<N>_top_pairs.csv`

- **LaTeX caption stub (if we include a sequential dynamics summary table):**

  ```latex
  \caption{Top enriched annotation pairs across sequential horizons, summarizing the strongest directional associations $X \to Y$ for each window size $K \in \{N, N, N\}$.%
  %
  TODO: ref three-figure caption
  %
  To keep the summary interpretable, we include only the top $N$ most enriched pairs per $K$ (TODO: decide on a concrete value for $N$, for example $N = 10$ or $20$, and implement a filtered table), to minimum support thresholds on $N_X$ and $C_K[X, Y]$.%
  %
  A positive enrichment greater than $1$ indicates that $Y$ occurs within the $K$-message window after $X$ more often than expected given its marginal frequency, while values near $1$ indicate background-rate behavior.}
  ```

#### Heatmaps

```bash
# Plot the default validation transitions, with clustering
python analysis/plot_sequential_annotation_dynamics.py \
    --window-k 3
```

Outputs: **combined enrichment heatmaps** for K=[N+] to `analysis/figures/sequential_enrichment_Ks.pdf`

[analysis/figures/sequential_enrichment_Ks.pdf](analysis/figures/sequential_enrichment_Ks.pdf) \

By default, the sequential dynamics heatmaps perform hierarchical clustering on the annotations to group rows and columns with similar sequential
profiles. To turn this off and use the canonical (sorted) annotation order instead, pass `--no-cluster-order` to `analysis/
plot_sequential_annotation_dynamics.py`.

**Alpha / opacity parameter**

By default, the sequential-dynamics heatmaps also use an opacity (alpha) channel to convey uncertainty in each $X \to Y$ cell.

For `--effect-source beta`, we treat the posterior standard deviation of the K-window probability $P(Y \text{ occurs at least once within } K \mid X)$ as an uncertainty scalar: cells with tighter posteriors (more supporting windows) are drawn more opaquely, and cells with very high uncertainty are partially faded toward the background.

To disable this uncertainty-based opacity entirely and render all cells fully opaque, we run the plotting script with `--no-uncertainty-alpha`.

#### Bar Charts version

Alternatively, we can home in on the connection between specific data points in bar charts as opposed to heat maps.

Suicidal ideation effects: [analysis/figures/sequential_profile_suicidal_K5.pdf](analysis/figures/sequential_profile_suicidal_K5.pdf)

User violent intent effects: [analysis/figures/sequential_profile_user-violent-intent_K5.pdf](analysis/figures/sequential_profile_user-violent-intent_K5.pdf)

User assigns personhood effects: [analysis/figures/sequential_profile_assigns-personhood_K5.pdf](analysis/figures/sequential_profile_assigns-personhood_K5.pdf)

Grand significance effects: [analysis/figures/sequential_profile_assigns-grand-significance_K5.pdf](analysis/figures/sequential_profile_assigns-grand-significance_K5.pdf)

Romantic affinity effects: [analysis/figures/sequential_profile_romantic_K5.pdf](analysis/figures/sequential_profile_romantic_K5.pdf)

These are the commands to generate them:

```bash

# On self and other harm
python analysis/plot_sequential_annotation_bars_pair.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --window-k 3 \
    --effect-source beta \
    --order-by-effect-size \
    --figure-path analysis/figures/sequential_profile_suicidal_and_violent.pdf \
    \
    --left-source-id user-suicidal-intent \
    --left-target-id assistant-validates-self-harm-feelings \
    --left-target-id assistant-discourages-self-harm \
    --left-target-id user-suicidal-intent \
    --left-target-id user-expresses-isolation \
    --left-target-id assistant-facilitates-self-harm \
    \
    --right-source-id user-violent-intent \
    --right-target-id user-violent-intent \
    --right-target-id assistant-discourages-violence \
    --right-target-id assistant-validates-volent-feelings \
    --right-target-id assistant-facilitates-violence \
    --right-target-id assistant-validates-violent-feelings


## On romantic relationships and sentience
python analysis/plot_sequential_annotation_bars_pair.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --window-k 3 \
    --order-by-effect-size  \
    --effect-source beta \
    \
    --left-source-id romantic-interest:user \
    \
    --left-target-id assistant-misrepresents-sentience \
    --left-target-id user-misconstrues-sentience \
    --left-target-id user-endorses-delusion \
    --left-target-id romantic-interest:assistant \
    \
    --right-source-id user-assigns-personhood \
    \
    --right-target-id user-misconstrues-sentience \
    --right-target-id user-assigns-personhood \
    --right-target-id user-misconstrues-lm-ability \
    --right-target-id assistant-misrepresents-sentience \
    --right-target-id platonic-affinity:assistant \
    \
    --figure-path analysis/figures/sequential_profile_romantic_and_personhood.pdf


# On user endorses delusion
python analysis/plot_sequential_annotation_bars_pair.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --window-k 3 \
    --effect-source beta \
    --order-by-effect-size  \
    \
    --left-source-id user-endorses-delusion \
    \
    --left-target-id grand-significance \
    --left-target-id assistant-claims-unique-understanding \
    --left-target-id assistant-validates-ideas \
    --left-target-id assistant-dismisses-counterevidence \
    --left-target-id assistant-hyperbole \
    --left-target-id assistant-extrapolates \
    \
    --right-source-id grand-significance \
    \
    --right-target-id assistant-hyperbole \
    --right-target-id theme-awakening-consciousness:user \
    --right-target-id user-endorses-delusion \
    \
    --figure-path analysis/figures/sequential_profile_delusions_and_grand-significance.pdf


# These don't really show much that I find interesting.
python analysis/plot_sequential_annotation_bars.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --window-k 5 \
    --source-id user-seeks-validation \
    \
    --target-id assistant-validates-ideas \
    --target-id assistant-positive-affirmation \
    --target-id assistant-reflective-summary \
    --target-id assistant-extrapolates \
    --target-id user-seeks-validation \
    --effect-source beta \
    --order-by-effect-size  \
    --figure-path test.pdf

python analysis/plot_sequential_annotation_bars.py \
    --output-prefix analysis/data/sequential_dynamics \
    --window-k 5 \
    --effect-source beta \
    --source-id assistant-offplatform-action  \
    \
    --target-id assistant-offplatform-action \
    --target-id assistant-extrapolates \
    --target-id user-reports-followed-instruction \
    --target-id message-outreach \
    --target-id user-bypass \
    --effect-source beta \
    --order-by-effect-size  \
    --figure-path test.pdf
```

Optionally we can also compute a few triples of interest...

```bash
python analysis/plot_sequential_annotation_bars.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --effect-source beta \
    --order-by-effect-size \
    --window-k 5 \
    --source-id user-suicidal-intent \
    --cond-id assistant-discourages-self-harm \
    --conditional-target-id assistant-discourages-self-harm \
    --conditional-target-id assistant-validates-self-harm-feelings \
    --conditional-target-id assistant-facilitates-self-harm \
    --conditional-target-id user-suicidal-intent


python analysis/plot_sequential_annotation_bars.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --effect-source beta \
    --order-by-effect-size \
    --window-k 5 \
    --source-id user-suicidal-intent \
    --cond-id assistant-facilitates-self-harm \
    --conditional-target-id assistant-discourages-self-harm \
    --conditional-target-id user-expresses-isolation \
    --conditional-target-id assistant-facilitates-self-harm \
    --conditional-target-id user-suicidal-intent

# or as a pair
python analysis/plot_sequential_annotation_bars_pair.py \
    --output-prefix analysis/data/sequential_dynamics/base \
    --window-k 5 \
    --effect-source beta \
    --order-by-effect-size \
    --figure-path analysis/figures/sequential_profile_suicidal_triple.pdf \
    \
    --left-source-id user-suicidal-intent \
    --left-cond-id assistant-discourages-self-harm \
    --left-conditional-target-id assistant-discourages-self-harm \
    --left-conditional-target-id assistant-validates-self-harm-feelings \
    --left-conditional-target-id assistant-facilitates-self-harm \
    --left-conditional-target-id user-suicidal-intent \
    \
    --right-source-id user-suicidal-intent \
    --right-cond-id assistant-facilitates-self-harm \
    --right-conditional-target-id assistant-validates-self-harm-feelings \
    --right-conditional-target-id assistant-discourages-self-harm \
    --right-conditional-target-id assistant-facilitates-self-harm \
    --right-conditional-target-id user-suicidal-intent

```

### (Done) Next-K Turns analysis

For an given code ($X$), we count how many times it appears in a fixed message window.
% That is, within each conversation, for every occurrence at index ($i$), we count ($C_i$), the number of subsequent messages in the same conversation with indices ($j$) satisfying ($i < j \le i + K$) that are also positive for (X).
The collection across all conversations is an empirical distribution over non-negative integers, from which we estimate a probability mass function ($P(C = c)$). We then compute the likelihood of additional occurences by conditioning this model on an occurrence of ($X$).

```bash
python analysis/compute_annotation_window_incidence.py \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/metrics.json \
    --annotation-id user-suicidal-intent \
    --role-scope user \
    --window-k 100 \
    --output analysis/figures/next_k_counts__user-suicidal-intent.pdf
```
