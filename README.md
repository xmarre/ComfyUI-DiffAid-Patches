# ComfyUI-DiffAid-Patches

Flux-first and SDXL-capable ComfyUI custom nodes that implement practical **Diff-Aid-style inference-time text-conditioning patches**.

This pack now contains two server-side nodes:

1. **Flux.2 Diff-Aid Sparse Patch**
2. **SDXL Diff-Aid Cross-Attention Patch**

## What this repository is and is not

This repository is a **usable inference-time ComfyUI patch pack**, not a paper-exact training reproduction.

The uploaded Diff-Aid paper introduces a learned Aid module that modulates text features **per block, per timestep, and per token**, and also reports a simpler **FLUX sparse enhancement** using blocks `{1, 15, 36, 41, 48}` with `alpha = 0.5` for those blocks and `0.0` elsewhere. The paper evaluates **FLUX** and **SD 3.5**, not SDXL. This pack therefore does two different things:

- for **Flux-family MMDiT models**, it implements the paper's practical sparse-enhancement path first
- for **SDXL-style cross-attention U-Nets**, it ports the same core idea into cross-attention context modulation as a best-effort architectural adaptation

So:

- **Flux node** = close to the paper's sparse inference strategy
- **SDXL node** = architectural port of the same principle, not paper-validated on SDXL

---

## Design summary

### Shared design principle

Both nodes preserve the same core behavior family:

```text
c' = c + c * alpha
```

where `alpha` is built from:

- a base `strength`
- optional timestep/sigma windowing
- optional token-position weighting

That matches the paper's central formulation of adaptive textual feature modulation while remaining simple enough to review and maintain.

### Flux node

The Flux node patches selected **MMDiT double/single blocks** through ComfyUI's `dit` block replacement hooks.

- double-stream blocks: modulates `txt`
- single-stream blocks: modulates the text-prefix slice inside the merged stream

Default preset:

```text
1, 15, 36, 41, 48
```

For a standard FLUX 57-block layout, this maps to:

- double blocks: `0, 14`
- single blocks: `16, 21, 28`

### SDXL node

The SDXL node patches **cross-attention (`attn2_patch`)** globally and then filters by stage/target at runtime.

It modulates the cross-attention context embeddings before the attention operation:

- `context_attn2`
- and `value_attn2` when shape-compatible

Supported filters:

- all stages
- only `input`
- only `middle`
- only `output`
- or explicit targets like:
  - `input:4`
  - `middle:0`
  - `output:7:1`

The `:1` suffix is the transformer-block index within a spatial transformer.

---

## Repository structure

```text
ComfyUI-DiffAid-Patches/
├── __init__.py
├── nodes.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> ComfyUI-DiffAid-Patches
```

No extra Python dependencies are required beyond a normal ComfyUI install.

Restart ComfyUI.

---

## Node reference

## 1) Flux.2 Diff-Aid Sparse Patch

**Node type:** `MODEL -> MODEL`

### Inputs

- `model` (`MODEL`)
- `enabled` (`BOOLEAN`)
- `block_preset`
  - `paper_sparse_flux`
  - `custom_combined_indices`
- `block_indices` (`STRING`)
  - only used for `custom_combined_indices`
  - comma-separated **1-based combined block indices**
- `strength` (`FLOAT`, default `0.5`)
- `sigma_start` (`FLOAT`, default `0.0`)
- `sigma_end` (`FLOAT`, default `1.0`)
- `sigma_ramp` (`FLOAT`, default `0.0`)
- `token_weight_mode`
  - `none`
  - `linear`
  - `exponential`
- `token_tail` (`FLOAT`, default `0.35`)

### Outputs

- patched `MODEL`
- summary `STRING`

### Recommended baseline settings

Closest to the sparse strategy described in the paper appendix:

- `block_preset = paper_sparse_flux`
- `strength = 0.5`
- `token_weight_mode = none`
- `sigma_start = 0.0`
- `sigma_end = 1.0`
- `sigma_ramp = 0.0`

### Example

```text
Flux model -> ModelSamplingFlux (optional / normal) -> Flux.2 Diff-Aid Sparse Patch -> sampler
```

---

## 2) SDXL Diff-Aid Cross-Attention Patch

**Node type:** `MODEL -> MODEL`

### Inputs

- `model` (`MODEL`)
- `enabled` (`BOOLEAN`)
- `stage_filter`
  - `all`
  - `input`
  - `middle`
  - `output`
- `block_targets` (`STRING`)
  - optional advanced override
  - examples:
    - `input:4`
    - `middle:0`
    - `output:7`
    - `output:7:1`
- `strength` (`FLOAT`, default `0.35`)
- `sigma_start` (`FLOAT`, default `0.0`)
- `sigma_end` (`FLOAT`, default `1.0`)
- `sigma_ramp` (`FLOAT`, default `0.0`)
- `token_weight_mode`
  - `none`
  - `linear`
  - `exponential`
- `token_tail` (`FLOAT`, default `0.35`)

### Outputs

- patched `MODEL`
- summary `STRING`

### Recommended first settings for SDXL

Start conservatively:

- `stage_filter = all`
- `block_targets = ""`
- `strength = 0.20` to `0.35`
- `token_weight_mode = linear`
- `token_tail = 0.35`
- full sigma window

### Example target strings

Patch only a few places:

```text
input:4, middle:0, output:7
```

Patch one specific transformer block inside an output spatial transformer:

```text
output:7:1
```

---

## Usage notes

### Which node should be used?

- **Flux / Flux.2 / Flux-family MMDiT**: use **Flux.2 Diff-Aid Sparse Patch**
- **SDXL-style UNet with cross-attention**: use **SDXL Diff-Aid Cross-Attention Patch**

### Why the Flux node is not SDXL-compatible

The Flux node expects a diffusion model that exposes:

- `double_blocks`
- `single_blocks`
- Flux-style merged-stream behavior

SDXL does not use that architecture. It uses a cross-attention UNet path instead, so the correct integration point is `attn2_patch`, not `dit` double/single block replacement.

### Patch ordering

Recommended order:

1. load model
2. apply any model-sampling wrapper node you normally use
3. apply this Diff-Aid patch node
4. connect patched `MODEL` to the sampler

---

## Assumptions

- ComfyUI build supports:
  - `set_model_unet_function_wrapper`
  - `set_model_attn2_patch`
  - `set_model_patch_replace(..., "dit", ...)`
  - `transformer_options`
- Flux-family models expose `double_blocks` and `single_blocks`
- SDXL-like models expose `input_blocks`, `middle_block`, and `output_blocks`
- sampling timestep is available via the model wrapper path

---

## Caveats and limitations

1. **Not full trained Diff-Aid.**
   The learned Aid MLP weights are not included here.

2. **SDXL version is a port, not a paper result.**
   The paper evaluates FLUX and SD 3.5, not SDXL.

3. **Token weighting is positional only.**
   It is not padding-aware or tokenizer-aware.

4. **The SDXL node is intentionally conservative.**
   Cross-attention U-Nets can be more sensitive to over-strengthening than the paper's FLUX sparse strategy.

5. **Other custom nodes can still conflict.**
   The Flux node chains existing `dit` replacement patches where present. The SDXL node appends a normal `attn2_patch`, which should compose with other patches, but any later node can still alter the same model.

6. **Runtime compatibility is only syntax-checked here.**
   This repo was not executed against a live local ComfyUI install in this environment.

---

## Known practical starting points

### Flux / Flux.2

```text
paper_sparse_flux
strength = 0.5
token_weight_mode = none
sigma_start = 0.0
sigma_end = 1.0
```

### SDXL

```text
stage_filter = all
block_targets = ""
strength = 0.25
token_weight_mode = linear
token_tail = 0.35
sigma_start = 0.0
sigma_end = 1.0
```

If SDXL starts overshooting prompt details or destabilizing style, lower `strength` first.
