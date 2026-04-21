# ComfyUI-DiffAid-Patches

ComfyUI custom nodes that apply **Diff-Aid-inspired inference-time text-conditioning patches** to supported diffusion models.

This repository is a **practical patch pack for ComfyUI inference**, not a paper-exact reproduction of the original Diff-Aid training method. It currently provides two nodes:

- **Flux.2 Diff-Aid Sparse Patch** — for Flux-family MMDiT models exposed through ComfyUI as `double_blocks` / `single_blocks`
- **SDXL Diff-Aid Cross-Attention Patch** — for SDXL-style cross-attention U-Nets, as an architectural adaptation of the same high-level idea

---

## Paper credit / reference

This repository is based on ideas from:

> **Binglei Li, Mengping Yang, Zhiyu Tan, Junping Zhang, Hao Li**  
> **Diff-Aid: Inference-time Adaptive Interaction Denoising for Rectified Text-to-Image Generation**  
> arXiv:2602.13585, 2026  
> https://arxiv.org/abs/2602.13585

The paper presents a lightweight **Aid** module for rectified text-to-image diffusion transformers that adaptively modulates textual features **per token, per block, and per denoising timestep**. It is trained with the backbone model frozen, and the learned modulation coefficients are used at inference time as a plug-in enhancement.

Please cite and credit the paper if you use this repository as part of experiments, implementations, ports, or derivative work.

### Suggested citation

```bibtex
@article{li2026diffaid,
  title={Diff-Aid: Inference-time Adaptive Interaction Denoising for Rectified Text-to-Image Generation},
  author={Li, Binglei and Yang, Mengping and Tan, Zhiyu and Zhang, Junping and Li, Hao},
  journal={arXiv preprint arXiv:2602.13585},
  year={2026}
}
```

---

## What the paper does

The paper’s central claim is that prompt adherence and image quality can be improved by strengthening text-image interaction **selectively**, rather than with a single global guidance scale.

Its Aid module learns a coefficient tensor `α` that depends on:

- the **current transformer block**
- the **current denoising timestep**
- the **current text token features**

and uses that to modulate the text features before they participate in attention:

```text
c̃ = c + c ⊙ α
```

The paper also adds:

- bounded modulation (`tanh`)
- a gating path to encourage sparsity
- regularization on `α`
- optional DPO / reward-based optimization during training

The result is a learned plug-in that is lightweight in parameter count but still adaptive and interpretable.

---

## What the paper reports

The paper evaluates Diff-Aid on **FLUX** and **SD 3.5** and reports consistent gains in prompt-following and image-quality metrics.

### Main reported results from the paper

On the paper’s reported benchmarks:

- **SD 3.5** improves from **9.31 → 9.48** on HPSv3 and from **0.72 → 0.77** on GenEval.
- **FLUX** improves from **10.42 → 10.71** on HPSv3 and from **0.68 → 0.70** on GenEval.
- The paper also reports gains on HPSv2, ImageReward, and Aesthetic Score for both baselines.

### Sparse-enhancement finding relevant to this repo

Besides the full learned method, the appendix reports a simpler **FLUX sparse enhancement** result by only boosting a small subset of blocks:

- selected FLUX blocks: **`{1, 15, 36, 41, 48}`**
- with **`α = 0.5`** on those blocks and **`0.0`** elsewhere

For that sparse variant, the paper reports the following FLUX results:

- baseline FLUX: **HPSv2 28.53 / HPSv3 10.42 / ImgRwd 0.89 / Aes 6.66**
- sparse enhancement: **28.61 / 10.57 / 0.98 / 6.71**
- full method: **28.80 / 10.84 / 0.95 / 6.76**

That appendix result is the main reason this repository starts with a **Flux sparse patch** instead of trying to fake the entire trained Aid pipeline.

---

## What this repository implements

This repository does **not** ship the paper’s trained Aid weights or the Aid MLP itself.

Instead, it implements a **reviewable inference-time approximation** inspired by the paper:

- select where to apply text-conditioning enhancement
- apply a modulation of the form:

```text
c' = c + c * α
```

- construct `α` from user-controlled terms:
  - a base `strength`
  - an optional normalized sigma / timestep window
  - optional token-position weighting

### Important caveat

This is not the same as the paper’s learned adaptive Aid module.

In this codebase:

- `α` is **not learned**
- `α` is **not inferred from the current text features by a trained Aid network**
- block specificity comes primarily from **which blocks are patched**, not from a separately learned block-wise coefficient function

So the correct description is:

- **paper**: trained adaptive block/timestep/token modulation
- **this repo**: practical inference-time modulation patch inspired by that idea, with a closer approximation for FLUX sparse enhancement and a best-effort SDXL port

---

## Why there are two nodes

The paper works on **rectified text-to-image diffusion transformers** and evaluates **FLUX** and **SD 3.5**.

Those architectures are not interchangeable in ComfyUI integration terms.

### Flux-family node

Flux-family models expose transformer block structure that can be patched directly through ComfyUI’s DiT replacement hooks. That makes a sparse block-selection implementation reasonable.

### SDXL node

SDXL is a **cross-attention U-Net**, not a Flux-style MMDiT with `double_blocks` and `single_blocks`. So the correct hook point is the UNet cross-attention path rather than Flux block replacement.

That means the SDXL node is **not paper-validated**. It is an architectural adaptation of the same broad principle: strengthen text-conditioning at specific inference locations.

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

No additional Python packages are required beyond a normal ComfyUI installation.

Restart ComfyUI after installation.

---

## Node 1: Flux.2 Diff-Aid Sparse Patch

**Type:** `MODEL -> MODEL`

This node targets **Flux-family models** that expose the expected diffusion transformer structure.

### What it patches

The node uses ComfyUI model patch hooks to modify selected Flux transformer blocks:

- **double-stream blocks**: modulates the `txt` tensor directly
- **single-stream blocks**: modulates the text-prefix region inside the merged stream when the required slice metadata is available

It also wraps the model call to capture the current timestep into transformer options so timestep-gated modulation can be applied inside the patch.

### Inputs

- `model` — input `MODEL`
- `enabled` — bypass switch
- `block_preset`
  - `paper_sparse_flux`
  - `custom_combined_indices`
- `block_indices` — comma-separated 1-based combined block indices, used for `custom_combined_indices`
- `strength` — modulation magnitude
- `sigma_start`, `sigma_end` — normalized active timestep window
- `sigma_ramp` — soft edge width for the sigma window
- `token_weight_mode`
  - `none`
  - `linear`
  - `exponential`
- `token_tail` — final-token weight for non-`none` token weighting
- `apply_single_stream` — whether selected single-stream blocks should also be patched

### Outputs

- patched `MODEL`
- summary `STRING`

### Paper sparse preset behavior

The paper’s sparse preset is authored for a canonical FLUX.1 block layout:

- **19 double blocks**
- **38 single blocks**
- paper combined indices: **`1, 15, 36, 41, 48`**

This repository automatically remaps those 1-based combined indices to the currently loaded Flux-family model.

For a canonical 57-block layout, the paper indices resolve to:

- double blocks: `0, 14`
- single blocks: `16, 21, 28`

However, **single-stream patching is disabled by default** in this node. So with the default settings, the practical effect of the paper preset is:

- patch the remapped **double-stream** subset first
- leave remapped single-stream blocks inactive unless `apply_single_stream = True`

That default was chosen because double-stream patching is the safer first approximation, while single-stream behavior is more architecture-sensitive.

### Recommended starting settings

Closest to the appendix sparse-enhancement direction:

- `block_preset = paper_sparse_flux`
- `strength = 0.5`
- `sigma_start = 0.0`
- `sigma_end = 1.0`
- `sigma_ramp = 0.0`
- `token_weight_mode = none`
- `apply_single_stream = False`

### Example chain

```text
Flux model -> ModelSamplingFlux (optional) -> Flux.2 Diff-Aid Sparse Patch -> sampler
```

---

## Node 2: SDXL Diff-Aid Cross-Attention Patch

**Type:** `MODEL -> MODEL`

This node targets **SDXL-style cross-attention U-Nets**.

### What it patches

The node installs an `attn2` patch and modulates cross-attention conditioning tensors before the attention operation:

- `context_attn2`
- `value_attn2` when shape-compatible

The patch can be applied:

- to all input/middle/output stages
- to one specific stage
- or to explicit block targets such as `input:4` or `output:7:1`

### Inputs

- `model` — input `MODEL`
- `enabled` — bypass switch
- `stage_filter`
  - `all`
  - `input`
  - `middle`
  - `output`
- `block_targets` — optional explicit targets
- `strength`
- `sigma_start`, `sigma_end`
- `sigma_ramp`
- `token_weight_mode`
  - `none`
  - `linear`
  - `exponential`
- `token_tail`

### Outputs

- patched `MODEL`
- summary `STRING`

### Example target strings

```text
input:4, middle:0, output:7
```

Specific transformer inside a spatial transformer:

```text
output:7:1
```

### Recommended first settings

Start conservatively because this is an out-of-paper port:

- `stage_filter = all`
- `block_targets = ""`
- `strength = 0.20` to `0.35`
- `token_weight_mode = linear`
- `token_tail = 0.35`
- full sigma window

### Example chain

```text
SDXL model -> SDXL Diff-Aid Cross-Attention Patch -> sampler
```

---

## How the modulation works in this repo

Both nodes use the same runtime modulation family:

```text
α = strength × time_gain × token_gain
c' = c + c * α
```

where:

- `time_gain` comes from the normalized sigma/timestep window
- `token_gain` comes from the selected token weighting mode
- the resulting `α` is broadcast over token embeddings

### Token weighting modes

- `none` — all tokens receive the same scaling
- `linear` — modulation decays linearly from early tokens to later tokens
- `exponential` — modulation decays exponentially toward later tokens

This was chosen to preserve the paper’s token-importance intuition in a simple, transparent form, even though it is not the paper’s learned per-token Aid network.

---

## Compatibility

### Use the Flux node when

- the model is Flux-family
- the loaded diffusion model exposes `double_blocks` and `single_blocks`

### Use the SDXL node when

- the model is an SDXL-style cross-attention U-Net
- the diffusion model exposes `input_blocks`, `middle_block`, and `output_blocks`
- it is **not** a Flux-family MMDiT

### Do not expect

- the Flux node to work on SDXL
- the SDXL node to behave like the paper’s SD 3.5 implementation
- either node to reproduce the paper’s trained Aid results exactly

---

## Limitations

1. **Not a paper-exact reproduction**  
   The paper trains lightweight Aid modules; this repo uses hand-constructed runtime modulation.

2. **No trained Aid weights included**  
   There is no shipped checkpoint corresponding to the paper.

3. **FLUX approximation is closer than SDXL**  
   The Flux sparse patch is directly motivated by the paper’s appendix sparse-enhancement result. The SDXL node is a best-effort architectural port.

4. **Single-stream Flux behavior is more sensitive**  
   That is why it is disabled by default in the sparse preset path.

5. **Results will be model- and workflow-dependent**  
   Especially for non-canonical Flux.2 variants and for SDXL workflows with additional model patches or LoRAs.

---

## Practical guidance

- Start with the **Flux sparse preset** before trying custom indices.
- Keep **single-stream patching off** unless you are deliberately testing it.
- For SDXL, start with **low strength** and broaden only if the effect is too weak.
- Treat this repo as an **experimental inference-time patch pack**, not as a claim of reproducing the paper’s published numbers.

---

## Acknowledgements

Credit for the original Diff-Aid method, motivation, analysis, and reported findings belongs to the paper authors:

- Binglei Li
- Mengping Yang
- Zhiyu Tan
- Junping Zhang
- Hao Li

This repository is an independent ComfyUI-oriented implementation inspired by that work.

---

## License

MIT
