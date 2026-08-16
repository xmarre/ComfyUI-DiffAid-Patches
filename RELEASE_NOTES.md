# ComfyUI-DiffAid-Patches v1.0.6

v1.0.6 is the first tagged GitHub release of the current Diff-Aid patch pack. It packages the four supported ComfyUI nodes, the completed MiniMax H3 integration, and the coordinated Spectrum MiniMax H3 compatibility contract that was validated in real native H3 generations.

## Included nodes

- **Flux-family Diff-Aid Sparse Patch** — sparse text-conditioning modulation for Flux-family transformer layouts.
- **WAN Diff-Aid Sparse Patch** — experimental sparse context modulation for native WAN-family video models.
- **MiniMax H3 Diff-Aid Sparse Patch** — experimental text-only modulation for native MiniMax H3 packed sequences, using `mod_segments` as the authoritative modality layout.
- **SDXL Diff-Aid Cross-Attention Patch** — an SDXL cross-attention adaptation of the same inference-time conditioning idea.

## MiniMax H3 + Spectrum interoperability

v1.0.6 publishes the producer half of the versioned Diff-Aid ↔ Spectrum MiniMax H3 external-patch contract. Spectrum MiniMax H3 v0.2.12+ is the matching consumer.

For an enabled nonzero H3 patch, Diff-Aid now publishes pure scalar/configuration metadata describing the resolved activation modulation and exposes the exact normalized sigma already derived by the H3 timestep wrapper. The contract does not retain tensors or model objects and does not add a second CUDA scalar synchronization.

This lets Spectrum distinguish Diff-Aid's deterministic text-activation modulation from LoRA/model-parameter perturbation, preserve cache identity across patched and unpatched configurations, and protect hard sigma-window regime changes without inflating Spectrum's calibrated model-aware patch-risk prior.

Validated workflow order:

```text
Load Diffusion Model
-> MiniMax H3 Diff-Aid Sparse Patch
-> Spectrum Apply MiniMax H3
-> guider / scheduler
```

### Coordinated real-runtime validation

Native MiniMax H3, ER-SDE, 20 steps, Spectrum `model_aware_mode=full`, Diff-Aid blocks `1,13,25,37,50`, strength `0.5`:

| Run | Diff-Aid window | Actual / forecast | Sampler time | Compatibility result |
|---|---|---:|---:|---|
| Spectrum control | inactive | 11 / 9 | 187.236 s | 0 model-aware extra NFEs |
| Diff-Aid full hard window | `[0.0,1.0]` | 11 / 9 | 189.001 s | 0 transitions, 0 forced actuals, 0 extra NFEs |
| Diff-Aid partial hard window | `[0.0,0.95]` | 11 / 9 | 184.946 s | transition detected at normalized sigma `0.947368` on an already-actual step; 0 extra NFEs |

All three runs preserved Spectrum's normal 11-actual / 9-forecast budget. In the matched multi-shot media test, the partial hard window also restored the intended cut between shots while retaining the observed prompt-adherence enhancement. Treat that media result as an empirical workflow result, not a universal default.

## Relationship to the Diff-Aid paper

This repository remains a **Diff-Aid-inspired inference-time patch pack**, not a paper-exact reproduction.

The paper's full method trains lightweight Aid modules that predict adaptive per-token modulation from text features with block and timestep awareness. This repository does not ship the trained Aid MLP or learned Aid weights. Its Flux sparse path is motivated by the paper appendix's explicit sparse-enhancement experiment, while the MiniMax H3, WAN, and SDXL ports are architecture-specific experimental adaptations.

The paper evaluates FLUX and SD 3.5 text-to-image generation and explicitly lists extension to modalities such as text-to-video as future work. No paper-level quality claim is made here for MiniMax H3 or WAN.

## Reliability and CI

The v1.0.6 release head passed the repository's automated checks before this release work:

- **55 pytest tests passed** on Python 3.12;
- Ruff compatibility checks passed;
- `compileall` passed;
- the Comfy registry publish for v1.0.6 completed successfully.

The repository now also uses the same release pattern as Spectrum MiniMax H3:

- GitHub releases are created only after the `tests` workflow succeeds for a push to `main`;
- the release targets the exact tested commit SHA;
- the version is read from `pyproject.toml`;
- the release contains `ComfyUI-DiffAid-Patches-v1.0.6.zip` plus `SHA256SUMS`;
- an existing tag/release is left untouched, making the workflow idempotent;
- the Comfy registry workflow pins its checkout and publish actions to reviewed commit SHAs and does not persist checkout credentials.

## Installation / update

For an existing checkout:

```bash
cd ComfyUI/custom_nodes/ComfyUI-DiffAid-Patches
git pull
```

Restart ComfyUI after updating.

For Spectrum + MiniMax H3 interoperability, use **ComfyUI-DiffAid-Patches v1.0.6+** with **ComfyUI-Spectrum-MiniMax-H3 v0.2.12+** and keep the patch order shown above.
