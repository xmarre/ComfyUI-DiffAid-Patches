# ComfyUI-DiffAid-Patches v1.0.7

v1.0.7 completes MiniMax H3 Native Masked and learned-latent refinement interoperability with the coordinated H3 Continuum / Spectrum workflow.

## Native Masked MiniMax H3 metadata

Core MiniMax H3 may represent target audio/video modulation metadata as per-row integer tensors when denoise masks are active. DiffAid now accepts that native representation without attempting to coerce the whole vector to one scalar modality tag.

- Scalar `mod_segments` behavior is unchanged.
- Valid 1-D integer per-row target metadata is accepted only when its length matches the packed segment range.
- Per-row target VIDEO/AUDIO metadata remains outside DiffAid's text-only modification scope.
- DiffAid continues modifying only scalar MiniMax H3 text modality-tag `1` segments.
- Malformed vector dtype/shape/length fails explicitly.
- Composition with pre-existing H3 block replacements remains covered.

## Marked sampler-2 refinement sigma semantics

Ordinary DiffAid sampling keeps its existing run-local normalized sigma semantics. A learned-latent second pass is different: it is a low-sigma continuation of the native H3 trajectory.

For a MODEL explicitly marked with the `h3_refinement` API-v1 contract, DiffAid now evaluates its sigma window against the producer-supplied full native H3 sigma reference instead of renormalizing the first refinement call to `1.0`.

This removes the artificial off->on transition that previously appeared in a three-step partial-denoise refinement with a `0.00..0.95` DiffAid window. The model function stays in the same DiffAid regime across the first and middle refinement calls, so Spectrum can forecast the middle step without weakening its hard-transition safety rule.

Malformed or absent refinement metadata falls back to the existing run-local behavior.

## Coordinated runtime validation

The complete CUDA workflow was validated with H3 Continuum exact `refine_state`, the learned MiniMax H3 latent upscaler/refiner, Spectrum, DiffAid and Untwisting RoPE metadata active together.

In the validated three-step sampler-2 run:

```text
step 0: actual
step 1: forecast
step 2: actual
```

The previous artificial DiffAid `false->true` transition no longer appeared at the middle step. Spectrum reported `2 actual + 1 forecast` per refined chunk and the resulting media quality was user-validated as impeccable.

## Validation

The release branch passes Ruff/compileall and the complete pytest suite covering scalar and vector H3 metadata, malformed-vector fail-closed behavior, normal run-local sigma normalization, marked-refinement full-trajectory normalization, patch runtime state consistency, and compatibility wrapper lifecycle.

This release is coordinated with Spectrum MiniMax H3 v0.2.17, H3 Continuum v3.4.1, and the integrated MiniMax H3 Latent Upscaler + Refine release.
