from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


PAPER_SPARSE_FLUX_COMBINED_BLOCKS: Tuple[int, ...] = (1, 15, 36, 41, 48)
PAPER_SOURCE_FLUX_DOUBLE_BLOCKS = 19
PAPER_SOURCE_FLUX_SINGLE_BLOCKS = 38
STATE_KEY = "diffaid_runtime_state"
_EPSILON = 1.0e-12


@dataclass(frozen=True)
class SharedConfig:
    strength: float
    sigma_start: float
    sigma_end: float
    sigma_ramp: float
    token_weight_mode: str
    token_tail: float
    cond_only: bool


@dataclass(frozen=True)
class FluxMappedBlocks:
    combined_1based: Tuple[int, ...]
    double_0based: Tuple[int, ...]
    single_0based: Tuple[int, ...]
    total_double: int
    total_single: int

    @property
    def total(self) -> int:
        return self.total_double + self.total_single


@dataclass(frozen=True)
class SdxlTargetSpec:
    stage: str
    block_number: int
    transformer_index: Optional[int] = None


def _dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_combined_block_indices(text: str) -> List[int]:
    parts = re.split(r"[\s,;:|]+", text.strip())
    out: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid block index '{part}'. Use a comma-separated list of positive integers.") from exc
        if value <= 0:
            raise ValueError(f"Invalid block index '{value}'. Indices are 1-based and must be >= 1.")
        out.append(value)
    if not out:
        raise ValueError("No block indices were provided.")
    return _dedupe_preserve_order(out)


def _parse_sdxl_targets(text: str) -> Tuple[SdxlTargetSpec, ...]:
    if not text.strip():
        return tuple()

    out: List[SdxlTargetSpec] = []
    for raw_part in re.split(r"[\n,;|]+", text.strip()):
        part = raw_part.strip()
        if not part:
            continue
        pieces = [p.strip().lower() for p in part.split(":") if p.strip()]
        if len(pieces) not in (2, 3):
            raise ValueError(f"Invalid SDXL target '{part}'. Use 'input:4', 'middle:0', or 'output:7:1'.")
        stage = pieces[0]
        if stage not in {"input", "middle", "output"}:
            raise ValueError(f"Invalid SDXL stage '{stage}' in target '{part}'.")
        try:
            block_number = int(pieces[1])
        except ValueError as exc:
            raise ValueError(f"Invalid block number in target '{part}'.") from exc
        if block_number < 0:
            raise ValueError(f"Invalid block number in target '{part}'. Must be >= 0.")
        transformer_index = None
        if len(pieces) == 3:
            try:
                transformer_index = int(pieces[2])
            except ValueError as exc:
                raise ValueError(f"Invalid transformer index in target '{part}'.") from exc
            if transformer_index < 0:
                raise ValueError(f"Invalid transformer index in target '{part}'. Must be >= 0.")
        out.append(SdxlTargetSpec(stage=stage, block_number=block_number, transformer_index=transformer_index))
    return tuple(out)


def _remap_stage_indices(indices_1based: Sequence[int], source_total: int, target_total: int) -> List[int]:
    if source_total <= 0:
        raise ValueError(f"source_total must be positive, got {source_total}.")
    if target_total <= 0:
        raise ValueError(f"target_total must be positive, got {target_total}.")

    out: List[int] = []
    for index_1based in indices_1based:
        if index_1based <= 0 or index_1based > source_total:
            raise ValueError(f"Stage-local block index {index_1based} is outside the source range 1..{source_total}.")
        if source_total == 1 or target_total == 1:
            out.append(1)
            continue
        scaled = round((index_1based - 1) * (target_total - 1) / (source_total - 1)) + 1
        out.append(int(scaled))
    return _dedupe_preserve_order(out)


def _paper_sparse_flux_source_double_indices() -> List[int]:
    return [index for index in PAPER_SPARSE_FLUX_COMBINED_BLOCKS if index <= PAPER_SOURCE_FLUX_DOUBLE_BLOCKS]


def _paper_sparse_flux_source_single_indices() -> List[int]:
    return [
        index - PAPER_SOURCE_FLUX_DOUBLE_BLOCKS
        for index in PAPER_SPARSE_FLUX_COMBINED_BLOCKS
        if index > PAPER_SOURCE_FLUX_DOUBLE_BLOCKS
    ]


def _remap_paper_sparse_flux_double_only_indices(total_double: int) -> List[int]:
    return _remap_stage_indices(
        _paper_sparse_flux_source_double_indices(),
        PAPER_SOURCE_FLUX_DOUBLE_BLOCKS,
        total_double,
    )


def _remap_paper_sparse_flux_indices(total_double: int, total_single: int) -> List[int]:
    mapped_double = _remap_paper_sparse_flux_double_only_indices(total_double)
    mapped_single_local = _remap_stage_indices(
        _paper_sparse_flux_source_single_indices(),
        PAPER_SOURCE_FLUX_SINGLE_BLOCKS,
        total_single,
    )
    mapped_single_combined = [total_double + index for index in mapped_single_local]
    return _dedupe_preserve_order([*mapped_double, *mapped_single_combined])


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sigma_window_gain(normalized_sigma: torch.Tensor, start: float, end: float, ramp: float) -> torch.Tensor:
    normalized_sigma = normalized_sigma.float().reshape(-1).clamp(0.0, 1.0)
    if start <= 0.0 and end >= 1.0:
        return torch.ones_like(normalized_sigma)
    if ramp <= 0.0:
        return ((normalized_sigma >= start) & (normalized_sigma <= end)).to(dtype=normalized_sigma.dtype)

    ramp = max(float(ramp), _EPSILON)
    left = _smoothstep01((normalized_sigma - (start - ramp)) / (2.0 * ramp))
    right = 1.0 - _smoothstep01((normalized_sigma - (end - ramp)) / (2.0 * ramp))
    return (left * right).clamp(0.0, 1.0)


def _token_weights(count: int, mode: str, tail: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if count <= 0:
        raise ValueError("Token count must be positive.")
    if mode == "none" or count == 1:
        return torch.ones((count,), device=device, dtype=dtype)

    positions = torch.linspace(0.0, 1.0, steps=count, device=device, dtype=dtype)
    tail = max(0.0, min(1.0, float(tail)))
    if mode == "linear":
        return 1.0 - (1.0 - tail) * positions
    if mode == "exponential":
        safe_tail = max(tail, 1.0e-6)
        return torch.exp(torch.log(torch.tensor(safe_tail, device=device, dtype=dtype)) * positions)
    raise ValueError(f"Unsupported token weighting mode: {mode}")


def _coerce_batch_value(value, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        t = value.to(device=device, dtype=dtype).reshape(-1)
    else:
        t = torch.tensor([float(value)], device=device, dtype=dtype)
    if t.numel() == 1:
        return t.repeat(batch_size)
    if t.numel() == batch_size:
        return t
    return t[:1].repeat(batch_size)


def _as_float_tensor(value, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32).reshape(-1)
    try:
        return torch.tensor([float(value)], device=device, dtype=torch.float32)
    except (TypeError, ValueError):
        return None


def _has_tensor_payload(value) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        return value.numel() > 0
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return len(value) > 0
    return True


def _detect_reference_latents(c: Dict, transformer_options: Dict) -> bool:
    for key in ("reference_latents", "ref_latents", "reference_image", "reference_images"):
        if _has_tensor_payload(c.get(key)):
            return True
    ref_tokens = transformer_options.get("reference_image_num_tokens", 0)
    try:
        return int(ref_tokens) > 0
    except (TypeError, ValueError):
        return False


def _cond_branch_gain(batch_size: int, config: SharedConfig, transformer_options: Optional[Dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not config.cond_only or transformer_options is None:
        return torch.ones((batch_size,), device=device, dtype=dtype)

    cond_or_uncond = transformer_options.get("cond_or_uncond", None)
    if cond_or_uncond is None:
        return torch.ones((batch_size,), device=device, dtype=dtype)

    if torch.is_tensor(cond_or_uncond):
        ids = cond_or_uncond.to(device=device).reshape(-1)
    else:
        try:
            ids = torch.tensor(list(cond_or_uncond), device=device)
        except TypeError:
            ids = torch.tensor([cond_or_uncond], device=device)
        ids = ids.reshape(-1)

    if ids.numel() == 0:
        return torch.ones((batch_size,), device=device, dtype=dtype)

    cond_rows = ids == 0
    if cond_rows.numel() == batch_size:
        mask = cond_rows
    elif batch_size % cond_rows.numel() == 0:
        mask = cond_rows.repeat_interleave(batch_size // cond_rows.numel())
    else:
        return torch.ones((batch_size,), device=device, dtype=dtype)
    return mask.to(dtype=dtype)


def _compute_alpha(reference_tensor: torch.Tensor, token_count: int, config: SharedConfig, transformer_options: Optional[Dict]) -> torch.Tensor:
    batch = reference_tensor.shape[0]
    transformer_options = transformer_options or {}
    normalized_sigma = transformer_options.get("normalized_sigma", None)
    if normalized_sigma is None:
        state = transformer_options.get(STATE_KEY, {}) or {}
        normalized_sigma = state.get("normalized_sigma", None)
    if normalized_sigma is None:
        normalized_sigma = 1.0

    sigma = _coerce_batch_value(normalized_sigma, batch, reference_tensor.device, reference_tensor.dtype)
    time_gain = _sigma_window_gain(sigma, config.sigma_start, config.sigma_end, config.sigma_ramp)
    token_gain = _token_weights(token_count, config.token_weight_mode, config.token_tail, reference_tensor.device, reference_tensor.dtype)
    branch_gain = _cond_branch_gain(batch, config, transformer_options, reference_tensor.device, reference_tensor.dtype)
    alpha = config.strength * time_gain[:, None, None] * branch_gain[:, None, None] * token_gain[None, :, None]
    return alpha.to(dtype=reference_tensor.dtype)


def _is_flux_family_model(model) -> bool:
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        return False
    required_attrs = ("double_blocks", "single_blocks", "txt_in", "forward_orig")
    return all(hasattr(diffusion_model, attr) for attr in required_attrs)


def _is_cross_attn_unet_model(model) -> bool:
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        return False
    return (
        hasattr(diffusion_model, "input_blocks")
        and hasattr(diffusion_model, "middle_block")
        and hasattr(diffusion_model, "output_blocks")
        and not hasattr(diffusion_model, "double_blocks")
    )


def _get_flux_block_counts(model) -> Tuple[int, int]:
    diffusion_model = model.get_model_object("diffusion_model")
    return len(diffusion_model.double_blocks), len(diffusion_model.single_blocks)


def _map_combined_indices_to_flux_stages(indices_1based: Sequence[int], total_double: int, total_single: int) -> FluxMappedBlocks:
    total = total_double + total_single
    double_indices: List[int] = []
    single_indices: List[int] = []
    for index_1based in indices_1based:
        if index_1based > total:
            raise ValueError(f"Block index {index_1based} is outside the model range 1..{total} (double={total_double}, single={total_single}).")
        if index_1based <= total_double:
            double_indices.append(index_1based - 1)
        else:
            single_indices.append(index_1based - total_double - 1)
    return FluxMappedBlocks(
        combined_1based=tuple(_dedupe_preserve_order(indices_1based)),
        double_0based=tuple(_dedupe_preserve_order(double_indices)),
        single_0based=tuple(_dedupe_preserve_order(single_indices)),
        total_double=total_double,
        total_single=total_single,
    )


class SharedTimestepWrapper(nn.Module):
    def __init__(self, existing_wrapper=None):
        super().__init__()
        self.existing_wrapper = existing_wrapper
        self._first_sigma_abs: Optional[float] = None
        self._last_sigma_abs: Optional[float] = None

    def _normalized_sigma(self, timestep, device: torch.device) -> torch.Tensor:
        t = _as_float_tensor(timestep, device=device)
        if t is None or t.numel() == 0:
            return torch.tensor([1.0], device=device, dtype=torch.float32)
        current = float(t.abs().max().item())
        reset_threshold = max(abs(self._last_sigma_abs or 0.0), 1.0) * 1.0e-4
        if self._first_sigma_abs is None or (self._last_sigma_abs is not None and current > self._last_sigma_abs + reset_threshold):
            self._first_sigma_abs = max(current, _EPSILON)
        self._last_sigma_abs = current
        if self._first_sigma_abs <= _EPSILON:
            return torch.zeros_like(t)
        return (t.abs() / self._first_sigma_abs).clamp(0.0, 1.0)

    def _inject_state(self, c: Dict, timestep) -> Dict:
        c = dict(c)
        transformer_options = dict(c.get("transformer_options", {}))
        state = dict(transformer_options.get(STATE_KEY, {}))
        device = timestep.device if torch.is_tensor(timestep) else torch.device("cpu")
        state["raw_timestep"] = timestep
        state["normalized_sigma"] = self._normalized_sigma(timestep, device=device)
        state["reference_latents"] = _detect_reference_latents(c, transformer_options)
        transformer_options[STATE_KEY] = state
        c["transformer_options"] = transformer_options
        return c

    def forward(self, model_function, params):
        def diffaid_model_function(input_x, timestep, **c):
            c = self._inject_state(c, timestep)
            return model_function(input_x, timestep, **c)

        if self.existing_wrapper is not None:
            return self.existing_wrapper(diffaid_model_function, params)
        return diffaid_model_function(params["input"], params["timestep"], **params["c"])


class FluxBlockReplacePatch(nn.Module):
    def __init__(self, stage: str, config: SharedConfig, existing_patch=None):
        super().__init__()
        self.stage = stage
        self.config = config
        self.existing_patch = existing_patch
        self._reported_reference_latents = False

    def _call_next(self, args: Dict, extra: Dict):
        if self.existing_patch is not None:
            return self.existing_patch(args, extra)
        return extra["original_block"](args)

    def _maybe_report_reference_latents(self, transformer_options: Dict):
        if self._reported_reference_latents:
            return
        state = transformer_options.get(STATE_KEY, {}) or {}
        if state.get("reference_latents", False):
            print("[ComfyUI-DiffAid-Patches] Reference latents detected; Diff-Aid modulation remains text-token-only.")
            self._reported_reference_latents = True

    def forward(self, args: Dict, extra: Dict):
        transformer_options = args.get("transformer_options", {}) or {}
        self._maybe_report_reference_latents(transformer_options)
        new_args = dict(args)

        if self.stage == "double":
            txt = args["txt"]
            alpha = _compute_alpha(txt, txt.shape[1], self.config, transformer_options)
            new_args["txt"] = txt + txt * alpha
            return self._call_next(new_args, extra)

        x = args["img"]
        img_slice = transformer_options.get("img_slice", None)
        if not img_slice:
            return self._call_next(new_args, extra)
        txt_len = int(img_slice[0])
        if txt_len <= 0:
            return self._call_next(new_args, extra)
        alpha = _compute_alpha(x[:, :txt_len, :], txt_len, self.config, transformer_options)
        prefix = x[:, :txt_len, :] + x[:, :txt_len, :] * alpha
        new_args["img"] = torch.cat((prefix, x[:, txt_len:, :]), dim=1)
        return self._call_next(new_args, extra)


class SDXLCrossAttentionPatch(nn.Module):
    def __init__(self, config: SharedConfig, stage_filter: str, targets: Tuple[SdxlTargetSpec, ...]):
        super().__init__()
        self.config = config
        self.stage_filter = stage_filter
        self.targets = targets

    def _matches(self, extra_options: Dict) -> bool:
        block = extra_options.get("block", None)
        if not block or not isinstance(block, tuple) or len(block) < 2:
            return False
        stage = str(block[0]).lower()
        number = int(block[1])
        transformer_index = int(extra_options.get("block_index", 0))
        if self.targets:
            for spec in self.targets:
                if stage != spec.stage or number != spec.block_number:
                    continue
                if spec.transformer_index is not None and transformer_index != spec.transformer_index:
                    continue
                return True
            return False
        if self.stage_filter == "all":
            return True
        return stage == self.stage_filter

    def forward(self, n: torch.Tensor, context_attn2: torch.Tensor, value_attn2: Optional[torch.Tensor], extra_options: Dict):
        if context_attn2 is None or not self._matches(extra_options):
            return n, context_attn2, value_attn2
        alpha = _compute_alpha(context_attn2, context_attn2.shape[1], self.config, extra_options)
        context_mod = context_attn2 + context_attn2 * alpha
        if value_attn2 is None or value_attn2.shape == context_attn2.shape:
            value_source = context_attn2 if value_attn2 is None else value_attn2
            value_mod = value_source + value_source * alpha
        else:
            value_mod = value_attn2
        return n, context_mod, value_mod


def _get_existing_flux_replace_patch(model, stage: str, stage_index: int):
    transformer_options = model.model_options.get("transformer_options", {})
    patches_replace = transformer_options.get("patches_replace", {})
    dit_patches = patches_replace.get("dit", {})
    key = ("double_block", stage_index) if stage == "double" else ("single_block", stage_index)
    return dit_patches.get(key)


def _fmt_ints(values: Sequence[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "-"


def _fmt_targets(values: Sequence[SdxlTargetSpec]) -> str:
    if not values:
        return "-"
    out = []
    for value in values:
        if value.transformer_index is None:
            out.append(f"{value.stage}:{value.block_number}")
        else:
            out.append(f"{value.stage}:{value.block_number}:{value.transformer_index}")
    return ", ".join(out)


class Flux2DiffAidSparsePatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "block_preset": (
                    ["paper_sparse_flux_double_only_safe", "paper_sparse_flux_full", "custom_combined_indices"],
                    {"default": "paper_sparse_flux_double_only_safe"},
                ),
                "block_indices": ("STRING", {"default": "1,15,36,41,48", "multiline": False, "advanced": True}),
                "strength": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "sigma_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_ramp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001, "advanced": True}),
                "token_weight_mode": (["none", "linear", "exponential"], {"default": "none"}),
                "token_tail": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "apply_single_stream": ("BOOLEAN", {"default": False, "advanced": True}),
                "cond_only": ("BOOLEAN", {"default": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "patch"
    CATEGORY = "model_patches/diffaid"

    def patch(
        self,
        model,
        enabled: bool,
        block_preset: str,
        block_indices: str,
        strength: float,
        sigma_start: float,
        sigma_end: float,
        sigma_ramp: float,
        token_weight_mode: str,
        token_tail: float,
        apply_single_stream: bool = False,
        cond_only: bool = True,
    ):
        if not enabled:
            return model, "disabled"
        if not _is_flux_family_model(model):
            raise ValueError("This node only supports Flux-family MODEL objects that expose double_blocks/single_blocks. Use the SDXL node for cross-attention UNet models.")
        if sigma_start > sigma_end:
            raise ValueError(f"sigma_start ({sigma_start}) must be <= sigma_end ({sigma_end}).")

        total_double, total_single = _get_flux_block_counts(model)
        if block_preset == "paper_sparse_flux_double_only_safe":
            requested_combined_indices = list(PAPER_SPARSE_FLUX_COMBINED_BLOCKS)
            combined_indices = _remap_paper_sparse_flux_double_only_indices(total_double)
            preset_name = f"paper_sparse_flux_double_only_safe_remapped_from_flux1(source_double={PAPER_SOURCE_FLUX_DOUBLE_BLOCKS})"
        elif block_preset in {"paper_sparse_flux_full", "paper_sparse_flux"}:
            requested_combined_indices = list(PAPER_SPARSE_FLUX_COMBINED_BLOCKS)
            combined_indices = _remap_paper_sparse_flux_indices(total_double, total_single)
            legacy = "legacy_alias_" if block_preset == "paper_sparse_flux" else ""
            preset_name = f"{legacy}paper_sparse_flux_full_remapped_from_flux1(source_double={PAPER_SOURCE_FLUX_DOUBLE_BLOCKS},source_single={PAPER_SOURCE_FLUX_SINGLE_BLOCKS})"
        else:
            requested_combined_indices = _parse_combined_block_indices(block_indices)
            combined_indices = requested_combined_indices
            preset_name = "custom_combined_indices"

        mapped = _map_combined_indices_to_flux_stages(combined_indices, total_double, total_single)
        active_single_0based = mapped.single_0based if apply_single_stream else tuple()
        active_combined_1based = tuple([index + 1 for index in mapped.double_0based] + [mapped.total_double + index + 1 for index in active_single_0based])
        config = SharedConfig(float(strength), float(sigma_start), float(sigma_end), float(sigma_ramp), token_weight_mode, float(token_tail), bool(cond_only))

        patched = model.clone()
        existing_model_wrapper = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(SharedTimestepWrapper(existing_wrapper=existing_model_wrapper))

        for block_index in mapped.double_0based:
            existing_patch = _get_existing_flux_replace_patch(patched, "double", block_index)
            patched.set_model_patch_replace(FluxBlockReplacePatch("double", config, existing_patch=existing_patch), "dit", "double_block", block_index)

        if apply_single_stream:
            for block_index in mapped.single_0based:
                existing_patch = _get_existing_flux_replace_patch(patched, "single", block_index)
                patched.set_model_patch_replace(FluxBlockReplacePatch("single", config, existing_patch=existing_patch), "dit", "single_block", block_index)

        inactive_single_0based = tuple(index for index in mapped.single_0based if index not in active_single_0based)
        summary = (
            f"flux_sparse preset={preset_name}; requested_combined_blocks=[{_fmt_ints(requested_combined_indices)}]; "
            f"mapped_combined_blocks=[{_fmt_ints(mapped.combined_1based)}]; active_combined_blocks=[{_fmt_ints(active_combined_1based)}]; "
            f"double_blocks_0based=[{_fmt_ints(mapped.double_0based)}]; single_blocks_0based=[{_fmt_ints(active_single_0based)}]; inactive_single_blocks_0based=[{_fmt_ints(inactive_single_0based)}]; "
            f"apply_single_stream={str(bool(apply_single_stream)).lower()}; cond_only={str(config.cond_only).lower()}; "
            f"strength={config.strength:.3f}; normalized_sigma_window=[{config.sigma_start:.3f}, {config.sigma_end:.3f}] ramp={config.sigma_ramp:.3f}; "
            f"token_weight_mode={config.token_weight_mode}; token_tail={config.token_tail:.3f}; reference_latents=runtime_detected_text_only; "
            f"model_total_blocks={mapped.total} (double={mapped.total_double}, single={mapped.total_single})"
        )
        print(f"[ComfyUI-DiffAid-Patches] {summary}")
        return patched, summary


class SDXLDiffAidCrossAttentionPatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "stage_filter": (["all", "input", "middle", "output"], {"default": "all"}),
                "block_targets": ("STRING", {"default": "", "multiline": False, "advanced": True}),
                "strength": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
                "sigma_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_ramp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001, "advanced": True}),
                "token_weight_mode": (["none", "linear", "exponential"], {"default": "linear"}),
                "token_tail": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "cond_only": ("BOOLEAN", {"default": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "patch"
    CATEGORY = "model_patches/diffaid"

    def patch(
        self,
        model,
        enabled: bool,
        stage_filter: str,
        block_targets: str,
        strength: float,
        sigma_start: float,
        sigma_end: float,
        sigma_ramp: float,
        token_weight_mode: str,
        token_tail: float,
        cond_only: bool = True,
    ):
        if not enabled:
            return model, "disabled"
        if not _is_cross_attn_unet_model(model):
            raise ValueError("This node expects an SDXL-style cross-attention UNet MODEL. It is not for Flux-family MMDiT models.")
        if sigma_start > sigma_end:
            raise ValueError(f"sigma_start ({sigma_start}) must be <= sigma_end ({sigma_end}).")

        targets = _parse_sdxl_targets(block_targets)
        config = SharedConfig(float(strength), float(sigma_start), float(sigma_end), float(sigma_ramp), token_weight_mode, float(token_tail), bool(cond_only))
        patched = model.clone()
        existing_model_wrapper = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(SharedTimestepWrapper(existing_wrapper=existing_model_wrapper))
        patched.set_model_attn2_patch(SDXLCrossAttentionPatch(config=config, stage_filter=stage_filter, targets=targets))

        summary = (
            f"sdxl_cross_attention stage_filter={stage_filter}; block_targets=[{_fmt_targets(targets)}]; "
            f"strength={config.strength:.3f}; normalized_sigma_window=[{config.sigma_start:.3f}, {config.sigma_end:.3f}] ramp={config.sigma_ramp:.3f}; "
            f"token_weight_mode={config.token_weight_mode}; token_tail={config.token_tail:.3f}; cond_only={str(config.cond_only).lower()}"
        )
        print(f"[ComfyUI-DiffAid-Patches] {summary}")
        return patched, summary


NODE_CLASS_MAPPINGS = {
    "Flux2DiffAidSparsePatch": Flux2DiffAidSparsePatchNode,
    "SDXLDiffAidCrossAttentionPatch": SDXLDiffAidCrossAttentionPatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2DiffAidSparsePatch": "Flux-family Diff-Aid Sparse Patch",
    "SDXLDiffAidCrossAttentionPatch": "SDXL Diff-Aid Cross-Attention Patch",
}
