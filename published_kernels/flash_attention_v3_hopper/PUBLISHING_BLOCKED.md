# FA3 Hopper — Publishing Blocked

**Kernel:** Flash Attention v3 forward (Hopper SM90, warp-specialized, TMA + persistent scheduling)
**Source:** `nvidia_cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py` (BSD-3-Clause)

## Blocker: Proprietary Dependency

`fmha.py` imports:

```python
from helpers import fmha_helpers as fmha_utils
```

`helpers/fmha_helpers.py` carries:

```
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
```

This file **cannot be redistributed** under any open-source license. Uploading it to a public
HuggingFace Hub repo would violate NVIDIA's terms.

## What fmha_helpers provides

The helpers module contains tile scheduling utilities, softmax helper functions, and
reference check routines used by fmha.py at both compile and run time. It is not a
thin testing wrapper — the kernel itself calls into it.

## How to unblock

**Option 1 — Replace with open equivalents**
Reimplement the helpers surface that `fmha.py` actually calls using only BSD-3 /
Apache-2.0 code. This requires auditing every `fmha_utils.*` call site in `fmha.py`
and writing clean replacements. The tile scheduler logic is the most complex piece.

**Option 2 — Convert to CUDA C++**
Translate the Python CuTe DSL kernel to CUDA C++ (same path as FA2). At that point
the Python helpers dependency disappears entirely and the C++ version can be packaged
with `build.toml` + `torch_binding.cpp`.

**Option 3 — Wait for NVIDIA to re-license**
Track the CUTLASS repo for a future helpers re-license to BSD-3. If `fmha_helpers.py`
is re-released under BSD-3, Option 1 becomes trivial.

## Candidate build.toml (for when unblocked)

```toml
[general]
name = "flash_attn_v3_hopper"
backends = ["cuda"]

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h",
]

[kernel.flash_attn_v3]
backend = "cuda"
src = ["kernel_src/flash_attention_v3.cu"]
depends = ["torch"]
cuda-capabilities = ["9.0"]   # Hopper H100 only
```
