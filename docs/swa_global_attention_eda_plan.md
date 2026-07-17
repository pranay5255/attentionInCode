# SWA, Global Attention, and MoE: Deep EDA and Visualization Plan

## Purpose

This document defines a publication-quality exploratory data analysis for the
sliding-window attention (SWA), global attention, and mixture-of-experts (MoE)
hardware experiments.

The analysis should not be a gallery of unrelated charts. It should form an
evidence ladder:

1. Establish what was measured and which observations are trustworthy.
2. Determine where SWA is faster or more memory-efficient than global attention.
3. Explain why the result changes across sequence lengths, sparsity levels,
   head geometries, execution modes, and GPU families.
4. Characterize runtime reliability, compilation cost, and failure regions.
5. Connect primitive attention and MoE measurements to model-level training
   decisions without making unsupported model-quality claims.

## Current evidence state

The current campaign is stored in:

`runs/swa_moe_hardware/20260717_120116_swa-moe-research-v2/`

The final manifest reports:

| Quantity | Value |
|---|---:|
| Planned cases | 14,976 |
| Executed cases | 4,416 |
| Successful cases | 4,174 |
| Failed cases | 242 |
| Budget-skipped cases | 10,560 |
| Coverage | 29.49% |
| Worker GPU-hours proxy | 8.0821 |
| Shards above the 5% drift threshold | 78 |

This dataset is useful, but it is not a complete factorial experiment. Missing
and failed observations are potentially informative and must not be silently
discarded. Coverage and reliability plots must appear before performance plots
to avoid survivor bias.

Additional audit constraints from the generated report are:

- The campaign did not achieve zero unexpected failures.
- GPU device counts and capabilities were verified.
- Distinct Modal task IDs were not observed for every replicated cell.
- Replicates are intended to represent independent fresh-container executions,
  but there are only two replicates per selected cell.
- Hierarchical confidence intervals should be treated as descriptive evidence,
  not high-powered inferential evidence.
- Model compositions are analytical system bounds. The campaign did not train a
  model or measure loss, accuracy, or downstream quality.

## Source tables

### `case_measurements.csv`

Use for per-replicate distributions, raw timing samples, tail latency, compile
time, memory usage, task IDs, and replicate diagnostics.

Important columns include:

- Identifiers: `case_id`, `cell_id`, `suite`, `hardware`, `replicate`,
  `runtime_profile`, `modal_task_id`, and `status`.
- Attention axes: `sequence_length`, `window`, `mode`, `batch_size`, `dtype`,
  `num_heads`, `head_dim`, `model_width`, and `block_size`.
- MoE axes: `tokens`, `num_experts`, `routing_variant`,
  `routed_experts_per_token`, `shared_experts_per_token`,
  `network_copies_per_token`, `routing_profile`, `capacity_factor`,
  `hidden_size`, and `intermediate_size`.
- Distributed axes: `world_size`, `collective`, `message_bytes_per_rank`, and
  `overlap`.
- Measurements: `compile_time_ms`, `first_call_ms`, `median_ms`, `p05_ms`,
  `p95_ms`, `cv_pct`, `peak_allocated_bytes`, `peak_reserved_bytes`,
  `tokens_per_second`, `useful_tflops`, `peak_efficiency_pct`,
  `algorithmic_flops`, `effective_bandwidth_gbps`, `gpu_ms_per_token`,
  `phase_median_ms`, `phase_share_pct`, `capacity`, and `feasibility`.

### `aggregate_measurements.csv`

Use as the primary source for matched comparisons and headline plots. It
contains replicate-aware medians and hierarchical bootstrap intervals.

Important columns include:

- `median_of_replicate_medians_ms`
- `hierarchical_ci95_low_ms`
- `hierarchical_ci95_high_ms`
- `successful_sample_replicates`
- `replicate_medians_ms`
- `tokens_per_second`
- `useful_tflops`
- `peak_efficiency_pct`
- `effective_bandwidth_gbps`
- `gpu_ms_per_token`
- `scaling_efficiency_pct`

### `environment_effects.csv`

Use for matched, one-variable runtime-profile comparisons against the baseline.
The principal response is `latency_effect_pct`.

### `model_compositions.csv`

Use for analytical model compositions across model depth, attention schedule,
FFN layout, and expert-parallel size.

The starred quantities `EGTime*`, `EGGPUTime*`, and `EGFLOPs*` are system bounds,
not empirical model-quality measurements.

### `manifest.json` and `report_data.json`

Use for planned coverage, budget selection, case and shard status, failures,
audit flags, drift, and machine-readable report information.

## Analysis units and matching rules

### Independence unit

The independent unit is the fresh-container replicate, not an individual timing
sample within a replicate. Inner timing iterations must not be treated as
independent observations.

### SWA versus global matching

Compare an SWA cell to a global-attention cell only when all other controlled
variables match:

- hardware;
- runtime profile;
- sequence length;
- forward or training mode;
- batch size;
- dtype;
- number of heads;
- head dimension;
- model width;
- block size;
- replicate policy.

Window size is the treatment variable. A global row is the matched baseline.

### Runtime-profile matching

Compare each runtime profile to `baseline` within the same workload cell. Do not
average profile effects over unmatched configurations.

### Hardware comparisons

Hardware is a moderator, not a numeric treatment. Prefer GPU-family facets or
matched cross-GPU ratios over bars that average all workloads within a GPU.

## Derived analysis variables

### Attention density

Define normalized attention density as:

```text
rho = min(window, sequence_length) / sequence_length
```

This makes a window comparable across sequence lengths. For example, a window
of 1,024 means something different at sequence lengths 2,048 and 16,384, while
their normalized densities are directly interpretable.

### Matched SWA speedup

```text
swa_speedup = global_median_ms / swa_median_ms
log2_swa_speedup = log2(swa_speedup)
```

- `swa_speedup > 1`: SWA is faster.
- `swa_speedup = 1`: no latency difference.
- `swa_speedup < 1`: global attention is faster.

`log2_swa_speedup` is preferred for diverging color scales because improvements
and regressions are symmetric around zero.

### Matched memory saving

```text
memory_saving = 1 - swa_peak_reserved_bytes / global_peak_reserved_bytes
```

Calculate the corresponding allocated-memory metric separately. Reserved and
allocated memory answer different questions and should not be mixed.

### Tail inflation and jitter

```text
tail_inflation = p95_ms / median_ms
jitter_span = (p95_ms - p05_ms) / median_ms
```

These metrics identify configurations that are fast on average but unpredictable
in training.

### Ideal and realized sparsity benefit

```text
ideal_flop_speedup = global_algorithmic_flops / swa_algorithmic_flops
realization_ratio = measured_swa_speedup / ideal_flop_speedup
```

The realization ratio should not be interpreted as a bounded hardware
efficiency percentage. Values can exceed one because cache behavior, launch
overhead, or other non-FLOP effects may also improve.

### Compile amortization

For an execution horizon of `N` calls:

```text
amortized_ms(N) =
    (compile_time_ms + first_call_ms + (N - 1) * median_ms) / N
```

Plot this at realistic horizons rather than ranking compiler profiles using
steady-state latency alone.

### Training penalty difference-in-differences

```text
training_penalty_delta =
    (swa_training_ms / swa_forward_ms)
    - (global_training_ms / global_forward_ms)
```

This isolates whether SWA has a distinct training penalty beyond the generic
cost of backward execution.

### Hardware portability

For matched configurations, derive:

```text
h100_over_a100 = a100_latency_ms / h100_latency_ms
b200_over_h100 = h100_latency_ms / b200_latency_ms
```

These ratios reveal which workload structures benefit disproportionately from a
newer GPU generation.

## Visual encoding conventions

- Use `log2(sequence_length)` on the primary X-axis.
- Use attention density `rho` on the primary Y-axis.
- Use a zero-centered diverging color scale for `log2_swa_speedup`.
- Use blue for improvement, red for regression, and neutral gray at no effect.
- Use gray fill for budget-skipped or unsupported cells.
- Use cross-hatching for failed cells.
- Use reduced opacity for drifted or low-confidence cells.
- Use black outlines for Pareto-optimal configurations.
- Use GPU family as facets or marker shapes when color already represents an
  outcome.
- Use logarithmic axes for sequence length, window, message bytes, and compile
  amortization horizon.
- Show raw observations beneath summaries whenever the number of points permits.
- Keep metric meaning consistent across the full report.

## Plot suite A: data integrity and coverage

### A1. Coverage alluvial

**Type:** Sankey or alluvial diagram.

**Flow:**

```text
planned -> suite -> hardware -> selected or budget-skipped -> succeeded or failed
```

**Question:** What fraction of the intended design was actually observed, and
where did selection or failure remove evidence?

The widths should represent case counts. Do not combine budget-skipped and
failed cases because they have different interpretations.

### A2. Status phase atlas

**Type:** faceted tile heatmap.

- X: sequence length.
- Y: attention density or window.
- Fill: succeeded, failed, budget-skipped, or unplanned.
- Facets: GPU family by head geometry, with separate forward and training pages.

**Question:** Is missingness concentrated in scientifically important regions?

This should replace coarse feasibility bars that collapse GPU, geometry, and
workload interactions.

### A3. Failure UpSet plot

**Type:** UpSet intersection chart.

Construct sets for conditions such as:

- GPU family;
- head dimension;
- forward or training;
- block size;
- runtime profile;
- sequence-length band;
- attention-density band.

**Question:** Which combinations jointly characterize failed cases?

### A4. Failure-root-cause flow

**Type:** Sankey diagram.

```text
suite -> hardware -> geometry or collective -> runtime profile -> error class
```

**Question:** Are failures caused by workload feasibility, hardware-specific
behavior, communication, compilation, or orchestration?

### A5. Shard execution timeline

**Type:** Gantt chart.

- X: wall-clock time.
- Y: shard, grouped by GPU family.
- Bar width: shard duration.
- Color: success, failure, drift, or budget-skipped.
- Annotation: scheduler stalls and campaign shutdown.

**Question:** Did the experimental environment or scheduler behavior change over
the run?

### A6. Runtime drift control chart

**Type:** statistical process-control chart.

- X: shard completion order.
- Y: observed-to-expected runtime ratio or normalized shard duration.
- Lines: reference value and drift threshold.
- Color: GPU family.

**Question:** Are timing measurements stationary enough to pool?

### A7. Replicate-task independence graph

**Type:** bipartite network.

- Left nodes: replicated cells.
- Right nodes: Modal task IDs.
- Edges: observed cell-to-task assignments.

**Question:** Where did multiple intended replicates share an execution identity?

## Plot suite B: distributions and measurement reliability

### B1. Latency raincloud plots

**Type:** half violin, box plot, and jittered raw points.

- X: log latency.
- Y: GPU and attention type.
- Facets: forward and training.
- Color: attention density band.

**Question:** Are distributions symmetric, multimodal, or dominated by outliers?

### B2. Empirical cumulative distribution functions

**Type:** ECDF.

- X: SWA speedup, GPU-ms/token, or tail inflation.
- Y: fraction of matched configurations at or below the value.
- Lines: GPU family.

Add vertical reference lines at speedups of 1.0, 1.1, 1.25, and 2.0.

**Question:** What fraction of configurations achieves a practically meaningful
benefit?

### B3. Bland-Altman replicate agreement

**Type:** Bland-Altman plot.

- X: mean latency of replicate 1 and replicate 2.
- Y: percentage difference between replicates.
- Color: GPU.
- Shape: suite.

**Question:** Does reproducibility deteriorate with workload size or suite?

### B4. Tail-risk funnel

**Type:** scatter or funnel plot.

- X: median latency.
- Y: `p95 / median`.
- Point size: CV percentage.
- Color: GPU family.
- Shape: forward or training.

**Question:** Which apparently fast configurations have unacceptable runtime
variance?

### B5. Ranked uncertainty forest

**Type:** interval or forest plot.

- X: matched SWA speedup with hierarchical interval.
- Y: selected configuration label.
- Reference line: speedup equal to one.

Rank configurations by the lower confidence bound, not the point estimate.

**Question:** Which improvements remain credible after measurement uncertainty?

### B6. Ranked replicate disagreement

**Type:** dumbbell or lollipop plot.

- X: latency.
- Y: cells ordered by absolute replicate disagreement.
- Endpoints: replicate 1 and replicate 2.

Do not plot an arbitrary first group of cells. Show the most unstable cells and
a stratified sample of stable cells.

## Plot suite C: primary SWA versus global analysis

### C1. Sparse Attention Advantage Atlas

This is the principal attention figure.

**Type:** faceted response heatmap.

- X: `log2(sequence_length)`.
- Y: attention density `rho`.
- Fill: `log2(global_latency / swa_latency)`.
- Columns: A100, H100, and B200.
- Rows: forward and training.
- Separate figure or page: head geometry.
- Contours: speedup equal to 1.0, 1.1, and 1.25.
- Hatching: failed or unsupported cells.
- Opacity: evidence confidence.

**Question:** In which hardware and workload regions does SWA provide a measured
benefit over matched global attention?

### C2. Crossover frontier

**Type:** phase-boundary line chart.

- X: attention density.
- Y: minimum sequence length at which the lower uncertainty bound shows SWA is
  faster.
- Lines: GPU family.
- Facets: head geometry by execution mode.

Use a second, stricter boundary for a minimum 10% speedup.

**Question:** At what context length does sparsity begin paying for its overhead?

### C3. Sparsity Dividend plot

**Type:** parity scatter.

- X: ideal FLOP speedup.
- Y: measured latency speedup.
- Diagonal: perfect FLOP realization.
- Bubble size: matched memory saving.
- Color: GPU family.
- Shape: forward or training.
- Outline: stable versus drifted.

**Question:** How much of the theoretical sparsity benefit is converted into
wall-clock benefit?

Important regimes are:

- theoretical and measured benefit;
- high theoretical benefit but poor hardware realization;
- measured gain larger than the FLOP model predicts;
- sparsity-induced regression.

### C4. Matched SWA/global dumbbells

**Type:** dumbbell plot.

- X: absolute median latency.
- Y: a curated set of representative matched configurations.
- Endpoints: global and SWA.
- Connection color: improvement or regression.

Choose representatives across short and long sequences, low and high density,
all GPUs, and forward and training. This plot supplies absolute context for the
relative heatmaps.

### C5. Speedup distribution by density

**Type:** ridgeline density plot.

- X: matched SWA speedup.
- Y: attention-density band.
- Facets: GPU and mode.

**Question:** Does increased sparsity shift the entire performance distribution,
or only create a few extreme wins?

### C6. Survivor-bias mirror

**Type:** paired-panel comparison.

- Left panel: successful cases only.
- Right panel: the same design with failed and skipped regions explicitly
  represented.
- Metric: speedup or feasibility by hardware and geometry.

**Question:** How much would the conclusion change if missingness were ignored?

## Plot suite D: interaction and ablation analysis

### D1. Head-geometry interaction

- X: head dimension.
- Y: matched SWA speedup.
- Lines: GPU family.
- Facets: execution mode, sequence-length band, and density band.

**Question:** Are SWA benefits tied to a particular shape rather than total model
width alone?

### D2. Block-size response

- X: block size.
- Y: latency, efficiency, or sparsity realization ratio.
- Lines: head geometry.
- Facets: GPU and execution mode.

**Question:** Which block sizes expose or hide hardware utilization problems?

### D3. Batch-efficiency curve

- X: batch size.
- Y: GPU-ms/token in the primary panel and tokens/second in a secondary panel.
- Lines: SWA and global attention.
- Facets: GPU, sequence-length band, and mode.

Do not use a dual Y-axis.

**Question:** Does batching improve throughput without increasing total GPU cost
per token?

### D4. Dtype matched-effect forest

- X: FP16-to-BF16 latency effect.
- Y: workload group.
- Intervals: replicate-aware uncertainty.
- Facets: GPU and execution mode.

**Question:** Is dtype sensitivity stable across hardware generations?

### D5. Training penalty map

- X: sequence length.
- Y: attention density.
- Fill: training-penalty difference-in-differences.
- Facets: GPU and geometry.

**Question:** Does SWA introduce a backward-pass penalty not visible in forward
benchmarks?

### D6. Runtime-profile effect caterpillar

- X: matched latency effect percentage.
- Y: runtime profile.
- Points: individual matched cells.
- Summary: median and interquartile interval.
- Facets: GPU and suite.

**Question:** Which environment or compiler changes generalize, and which have
workload-dependent effects?

## Plot suite E: hardware mechanisms and trainer-facing tradeoffs

### E1. Memory versus GPU-time Pareto frontier

**Type:** Pareto scatter.

- X: peak reserved memory in GiB.
- Y: GPU-ms/token, with lower values preferred.
- Color: attention density.
- Size: sequence length.
- Shape: SWA or global.
- Facets: GPU family.
- Outline: nondominated frontier.

**Question:** Which configurations are simultaneously memory-efficient and
compute-efficient?

### E2. Attention efficiency envelope

- X: algorithmic FLOPs per token or attention density.
- Y: useful TFLOPS or peak-efficiency percentage.
- Color: GPU.
- Shape: attention type.
- Facets: mode.

This should not be called a roofline unless an accurate estimate of bytes moved
is available. Peak allocated memory is not memory traffic.

**Question:** Which regimes fail to convert available arithmetic work into GPU
utilization?

### E3. Hardware portability quadrant

- X: `log2(A100 latency / H100 latency)`.
- Y: `log2(H100 latency / B200 latency)`.
- Color: attention density.
- Size: sequence length.
- Shape: forward or training.

Add marginal marks for configurations that failed on one GPU and therefore
cannot form a complete matched triplet.

**Question:** Which workload structures disproportionately benefit from newer
GPU generations?

### E4. Cost-of-compilation quadrant

- X: compile time divided by steady-state latency.
- Y: steady-state speedup over the baseline profile.
- Color: runtime profile.
- Shape: GPU.

**Question:** Which compiler profiles deliver meaningful steady-state wins
without excessive startup cost?

### E5. Compile amortization curves

- X: number of invocations on a logarithmic scale.
- Y: amortized latency per invocation.
- Lines: runtime profiles.
- Facets: GPU and representative workload.
- Vertical markers: 10, 100, 1,000, and 10,000 invocations.

**Question:** At a realistic training horizon, which runtime profile is actually
cheapest?

### E6. Configuration regret map

- X: sequence-length and density cell.
- Y: GPU family.
- Fill: performance gap from the best supported configuration for that workload.

**Question:** How costly is a portable default compared with per-GPU tuning?

## Plot suite F: MoE and distributed communication

### F1. Strong-scaling efficiency

- X: world size.
- Y: scaling efficiency percentage.
- Lines: routing variant or collective.
- Facets: hardware and workload size.

Raw throughput should be shown in a companion panel, not substituted for scaling
efficiency.

**Question:** At what parallel size do additional GPUs stop producing
proportional gains?

### F2. Communication saturation curve

- X: message bytes per rank on a logarithmic scale.
- Y: effective bandwidth in GB/s.
- Lines: world size.
- Color: GPU family.
- Shape: overlap enabled or disabled.

**Question:** Where do collectives transition from launch-limited to
bandwidth-limited behavior?

### F3. Communication phase ternary

**Type:** ternary plot.

- Coordinates: packing share, communication share, and expert-compute share.
- Point size: token count or total latency.
- Color: world size.
- Shape: collective or overlap policy.

**Question:** Which cells are compute-bound, communication-bound, or packing-
bound?

### F4. Communication stacked decomposition

- X: world size or message size.
- Y: total latency.
- Stacks: packing, forward collective, expert compute, return collective, and
  other measured phases.
- Facets: GPU and routing variant.

**Question:** Which phase causes the loss of scaling efficiency?

### F5. Capacity phase diagram

- X: number of experts.
- Y: capacity factor or tokens per expert.
- Fill: feasibility, overflow, or throughput.
- Contours: equal latency or equal throughput.
- Facets: routing variant and GPU.

**Question:** Where does expert capacity move from inadequate to wasteful?

### F6. Routing-regret heatmap

- X: expert count.
- Y: expert-parallel size or world size.
- Fill: performance difference from the best routing strategy at the matched
  workload.
- Facets: hardware and token count.

**Question:** Is a single routing strategy robust across scales?

### F7. Communication-overlap effect forest

- X: matched overlap speedup with uncertainty.
- Y: message-size and world-size group.
- Facets: GPU family and collective.

**Question:** When does communication overlap hide useful latency rather than
add coordination overhead?

### F8. Imbalance-throughput frontier

- X: load-imbalance or capacity-utilization measure.
- Y: tokens per second.
- Size: dropped or excess-capacity proxy, if available.
- Color: routing profile.

**Question:** How directly does routing imbalance translate into throughput
loss?

## Plot suite G: model- and trainer-level synthesis

### G1. Schedule component waterfall

- X: `all_global`, `1:1`, `3:1`, `5:1`, `7:1`, `15:1`, and `all_swa`.
- Y: predicted model-step time.
- Stacks: global attention, SWA, dense FFN, MoE compute, and communication.
- Facets: model depth, FFN layout, and expert-parallel size.

**Question:** Which primitive component explains each schedule's predicted
advantage?

### G2. Model-composition Pareto frontier

- X: predicted GPU time per step.
- Y: algorithmic FLOPs.
- Color: attention schedule.
- Shape: FFN layout.
- Size: model depth.
- Facets: hardware or expert-parallel size.

Label only nondominated configurations to avoid clutter.

**Question:** Which systems compositions dominate others on both compute and GPU
time?

### G3. Amdahl landscape

- X: baseline fraction of model-step time spent in attention.
- Y: primitive attention speedup.
- Fill or contours: maximum end-to-end model speedup.
- Overlay: analytical model compositions.

**Question:** How much model-level improvement is possible even if the attention
kernel becomes much faster?

### G4. Trainer recipe map

- X: context length.
- Y: global-layer fraction or SWA density.
- Fill: best measured systems candidate.
- Opacity: evidence confidence and coverage.
- Hatching: unsupported or failure-prone region.
- Facets: GPU family.

This is a systems recommendation map, not a model-quality recommendation map.

### G5. Sensitivity tornado

**Type:** tornado chart based on matched composition changes.

- Y: design decision such as schedule, FFN layout, expert-parallel size, or
  depth.
- X: resulting percentage change in model-step time or GPU time.

**Question:** Which model-level choice has the largest systems consequence?

## Advanced diagnostic appendix

### Predictive feature importance

Fit a transparent cross-validated model for log latency and report permutation
importance. Group train/test splits by `cell_id` so replicates do not leak across
folds. Interpret importance as predictive, not causal.

### Accumulated local effects

Use ALE plots rather than naive partial dependence when sequence length, window,
and density are strongly dependent. Recommended effects are:

- sequence length;
- density;
- head dimension;
- block size;
- batch size;
- sequence-by-density interaction;
- GPU-by-geometry interaction.

### Behavioral configuration embedding

Use PCA or UMAP on standardized behavioral outputs such as latency, memory,
tail inflation, compile ratio, and efficiency. Color by failure, GPU, or
attention regime. Treat the embedding as an anomaly-discovery tool, not
scientific proof.

### Residual phase map

Fit a simple scaling law for log latency, then plot residuals over sequence
length and density. Large residual regions can expose kernel boundaries,
unmodeled geometry effects, or measurement anomalies.

### GPU-time Lorenz curve

Plot cumulative configuration families against cumulative worker GPU-time. This
shows whether a small part of the experimental matrix consumed most of the
budget and can inform future sampling.

## Recommended main-paper figure set

If only twelve figures are implemented initially, use this order:

1. Coverage alluvial.
2. Status phase atlas.
3. Failure UpSet and root-cause summary.
4. Runtime drift and shard timeline.
5. Latency raincloud or ECDF.
6. Bland-Altman replicate agreement.
7. Sparse Attention Advantage Atlas.
8. Crossover frontier.
9. Sparsity Dividend plot.
10. Memory versus GPU-time Pareto frontier.
11. MoE strong-scaling and communication decomposition.
12. Model-level Amdahl landscape and composition frontier.

This order creates a defensible narrative from evidence quality to primitive
behavior and finally to systems implications.

## Composite hero figure

Create a six-panel figure titled:

**Sparse Attention-MoE Hardware Efficiency Atlas**

Recommended layout:

| Panel | Content | Role |
|---|---|---|
| A | Coverage alluvial and failure count | Establishes evidence support |
| B | Sequence-by-density SWA speedup map | Main algorithmic result |
| C | Ideal FLOP benefit versus measured speedup | Explains hardware realization |
| D | Memory versus GPU-time Pareto frontier | Shows trainer-facing tradeoff |
| E | MoE communication ternary or saturation plot | Shows distributed mechanism |
| F | Amdahl landscape with model compositions | Bounds end-to-end implications |

This should be a coordinated multi-panel figure, not a single overloaded 3D
chart. Attention and MoE should be connected by the narrative and the final
model composition, not placed on incompatible axes.

## Interactive exploration design

An interactive dashboard can complement the paper figures. It should expose:

- GPU family selector;
- forward/training selector;
- head geometry selector;
- dtype selector;
- batch and block-size selectors;
- runtime-profile selector;
- sequence and density ranges;
- status and drift filters;
- metric selector for latency, speedup, memory, tail, efficiency, and
  GPU-ms/token;
- click-through from an aggregate cell to its replicate timings and audit data.

The default dashboard view should be the Sparse Attention Advantage Atlas, not
a table of raw rows.

## Statistical and reporting requirements

1. Use replicate-aware aggregates for primary conclusions.
2. Keep failed and budget-skipped cells visible.
3. Match treatment and baseline cells on all controlled variables.
4. Do not pool hardware families before examining interactions.
5. Show uncertainty for ranked comparisons and crossover boundaries.
6. Prefer effect sizes and practical thresholds over p-values alone.
7. If formal hypothesis tests are added, correct for multiple comparisons.
8. Distinguish peak allocated memory from peak reserved memory.
9. Distinguish steady-state speed from amortized compilation cost.
10. Distinguish theoretical FLOP reduction from measured wall-clock speedup.
11. Do not call an efficiency-envelope plot a roofline without traffic data.
12. Do not present UMAP, feature importance, or partial-dependence plots as
    causal evidence.
13. Label all model-composition quantities as analytical, starred system bounds.
14. Do not make architecture-quality claims without loss or downstream
    evaluation.

## Graph types to avoid or constrain

- Avoid pie charts for configuration coverage.
- Avoid radar charts except for a tiny executive comparison of at most three or
  four configurations.
- Avoid dual Y-axes.
- Avoid static 3D surfaces; an interactive surface may be offered only as a
  supplementary explorer.
- Avoid averaging latency across incompatible sequence lengths or geometries.
- Avoid spaghetti plots containing every configuration line.
- Avoid reporting only the best configuration without showing its uncertainty,
  feasibility, and coverage neighborhood.
- Avoid treating failed configurations as if they never existed.

## Implementation sequence

### Phase 1: analysis-ready tables

1. Load manifest, case, aggregate, environment, and model-composition data.
2. Normalize hardware and status labels.
3. Derive attention density and matched-cell keys.
4. Match SWA cells to their global baselines.
5. Derive speedup, memory saving, tail, jitter, compile, and portability
   variables.
6. Add failure, drift, and coverage annotations to every analysis cell.
7. Validate that each comparison changes only its intended treatment variable.

### Phase 2: evidence audit

Implement plots A1 through A7 and B3 before drawing performance conclusions.
Produce a machine-readable table of excluded or unmatched cells with reasons.

### Phase 3: primary attention results

Implement the Advantage Atlas, crossover frontier, Sparsity Dividend plot,
matched dumbbells, and the core interaction plots.

### Phase 4: trainer-facing systems analysis

Implement memory/GPU-time Pareto fronts, compile amortization, tail-risk views,
and hardware portability.

### Phase 5: MoE and communication

Implement scaling efficiency, saturation, ternary phase composition, capacity,
and routing-regret plots.

### Phase 6: model-level synthesis

Implement schedule waterfalls, composition Pareto fronts, the Amdahl landscape,
and the systems-only trainer recipe map.

### Phase 7: final report packaging

1. Export vector PDF or SVG versions for paper figures.
2. Export high-resolution PNG versions for quick review.
3. Save the exact plotted data beside every figure.
4. Save plot configuration and filter metadata.
5. Generate a figure index containing title, research question, source table,
   filters, and interpretation limits.

## Reviewer-facing claims this EDA can support

Subject to the observed data and uncertainty, the analysis can support claims
about:

- where SWA produces real latency or memory benefits relative to matched global
  attention;
- how the crossover depends on context length, density, head geometry, mode,
  and GPU generation;
- how much theoretical sparsity benefit is realized in wall-clock execution;
- which runtime profiles amortize under realistic invocation horizons;
- which MoE workloads become communication- or capacity-limited;
- which analytical model compositions are systems-Pareto-optimal under the
  measured primitive bounds.

It cannot by itself support claims that:

- SWA improves model quality;
- an attention schedule preserves equal loss;
- analytical model-step compositions equal end-to-end training measurements;
- one GPU or runtime profile is universally superior outside the covered matrix.

## External methodological context

- [Long-Context Attention Benchmark](https://arxiv.org/abs/2510.17896)
- [AttentionEngine](https://arxiv.org/abs/2502.15349)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)
- [MoE Parallel Folding](https://arxiv.org/abs/2504.14960)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [BigBird](https://arxiv.org/abs/2007.14062)
