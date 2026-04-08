# Base Experiments (Phases 1-3)

This repo now has a clean "baseline" path per implementation without relying on per-folder `experiments/` suites.

## What "base" means

Each phase runs its built-in default runtime cases (`DEFAULT_CASES`):
- one dense forward case
- one causal forward case
- reference validation enabled

## Phase differences

- `01_flash_attention_v2_ampere_cute_dsl`
  - CuTe DSL Python kernel wrapper (`flash_attention_v2.py`)
  - Runtime: `fa2_cute_runtime.py`
  - GPU target: Ampere+ (`A100` in Modal app)
  - Defaults: `bf16`, `B=1`, `H=8`, `Sq=Sk=256`, `D=128`

- `02_fused_mha_ampere_cpp`
  - Compiled CUTLASS C++ binary wrapper (`fused_mha.py`)
  - Runtime: `fmha_cpp_runtime.py`
  - GPU target: Ampere+ (`A100` in Modal app)
  - Defaults: `f16`, `B=16`, `H=12`, `Sq=Sk=1024`, `D=64`

- `03_flash_attention_v3_hopper_cute_dsl`
  - Hopper CuTe DSL Python kernel wrapper (`flash_attention_v3.py`)
  - Runtime: `fa3_cute_runtime.py`
  - GPU target: Hopper+ (`H100` in Modal app)
  - Defaults: `f16`, `B=4`, `H=8`, `Sq=Sk=1024`, `D=64`

These defaults are intentionally not apples-to-apples. They are implementation-native sanity/perf baselines.

## Run baseline experiments

Run one phase (Modal Python entrypoints):

```bash
uv run modal run base_experiments/modal_base_exp_01_fa2_ampere.py
uv run modal run base_experiments/modal_base_exp_02_fmha_cpp_ampere.py
uv run modal run base_experiments/modal_base_exp_03_fa3_hopper.py
```

Run all three sequentially:

```bash
uv run modal run base_experiments/modal_base_exp_01_fa2_ampere.py
uv run modal run base_experiments/modal_base_exp_02_fmha_cpp_ampere.py
uv run modal run base_experiments/modal_base_exp_03_fa3_hopper.py
```

## Direct Modal commands (equivalent)

```bash
uv run modal run base_experiments/modal_base_exp_01_fa2_ampere.py
uv run modal run base_experiments/modal_base_exp_02_fmha_cpp_ampere.py
uv run modal run base_experiments/modal_base_exp_03_fa3_hopper.py
```
