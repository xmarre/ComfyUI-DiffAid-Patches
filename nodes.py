from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


PAPER_SPARSE_FLUX_COMBINED_BLOCKS: Tuple[int, ...] = (1, 15, 36, 41, 48)
PAPER_SOURCE_FLUX_DOUBLE_BLOCKS = 19
PAPER_SOURCE_FLUX_SINGLE_BLOCKS = 38
PAPER_SOURCE_FLUX_TOTAL_BLOCKS = PAPER_SOURCE_FLUX_DOUBLE_BLOCKS + PAPER_SOURCE_FLUX_SINGLE_BLOCKS
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
class WanMappedBlocks:
    requested_1based: Tuple[int, ...]
    mapped_1based: Tuple[int, ...]
    mapped_0based: Tuple[int, ...]
    total: int


@dataclass(frozen=True)
class MiniMaxH3MappedBlocks:
    requested_1based: Tuple[int, ...]
    mapped_0based: Tuple[int, ...]
    total: int


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


def _remap_paper_sparse_flux_to_single_block_list(total_blocks: int) -> List[int]:
    return _remap_stage_indices(
        PAPER_SPARSE_FLUX_COMBINED_BLOCKS,
        PAPER_SOURCE_FLUX_TOTAL_BLOCKS,
        total_blocks,
    )


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sigma_window_gain(normalized_sigma: torch.Tensor, start: float, end: float, ramp: float) -> torch.Tensor:
    normalized_sigma = normalized_sigma.float().reshape(-1).clamp(0.0, 1.0)
    if start <= 0.0 and end >= 1.0:
        return torch.ones_like(normalized_sigma)
    if ramp <= 0.0:
        return ((normalized_sigma >= start) & (normalized_sigma <= end)).to(dtype=normalized_sigma.dtype)

    # Keep sigma_start/sigma_end as the full-strength window boundaries.
    # Ramp only softens the shoulders outside that window, so a boundary that
    # touches 0.0 or 1.0 is not accidentally reduced to half strength.
    ramp = max(float(ramp), _EPSILON)
    if start <= 0.0:
        left = torch.ones_like(normalized_sigma)
    else:
        left_edge = max(0.0, start - ramp)
        left_width = max(start - left_edge, _EPSILON)
        left = _smoothstep01((normalized_sigma - left_edge) / left_width)

    if end >= 1.0:
        right = torch.ones_like(normalized_sigma)
    else:
        right_edge = min(1.0, end + ramp)
        right_width = max(right_edge - end, _EPSILON)
        right = 1.0 - _smoothstep01((normalized_sigma - end) / right_width)

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


def _max_abs_positive(value, device: torch.device) -> Optional[float]:
    t = _as_float_tensor(value, device=device)
    if t is None or t.numel() == 0:
        return None
    sigma_abs = float(t.abs().max().item())
    if sigma_abs <= _EPSILON:
        return None
    return sigma_abs


def _has_tensor_payload(value) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        return value.numel() > 0
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return len(value) > 0
    return True


def _detect_reference_latents(c: Dict, transformer_options: Dict) -> bool:
    for key in (
        "reference_latents",
        "ref_latents",
        "reference_latent",
        "ref_latent",
        "reference_image",
        "reference_images",
    ):
        if _has_tensor_payload(c.get(key)):
            return True
    ref_tokens = transformer_options.get("reference_image_num_tokens", 0)
    try:
        return int(ref_tokens) > 0
    except (TypeError, ValueError):
        return False


def _sequence_length(value) -> Optional[int]:
    if not torch.is_tensor(value):
        return None
    # WAN image context features are expected to be batched sequences
    # such as [batch, tokens, channels]. A 2D tensor is ambiguous because
    # shape[-2] would usually be the batch dimension, not a token count.
    if value.ndim < 3:
        return None
    try:
        return int(value.shape[-2])
    except (TypeError, ValueError):
        return None


def _detect_wan_context_image_tokens(c: Dict, transformer_options: Dict) -> int:
    for key in ("wan_context_img_len", "context_img_len", "context_image_token_count", "image_context_token_count"):
        value = transformer_options.get(key, None)
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count

    for key in ("clip_fea", "clip_vision_output", "image_embeds"):
        count = _sequence_length(c.get(key))
        if count is not None and count > 0:
            return count
    return 0


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


def _iter_wrapper_children(obj: Any):
    for attr in ("model", "diffusion_model", "inner_model", "module", "wrapped_model", "_orig_mod"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            yield attr, child


def _looks_like_wan_model(inner: Any) -> bool:
    return (
        callable(getattr(inner, "forward_orig", None))
        and hasattr(inner, "blocks")
        and hasattr(inner, "patch_embedding")
        and hasattr(inner, "head")
        and not hasattr(inner, "double_blocks")
        and not hasattr(inner, "input_blocks")
    )


def _locate_wan_like_descendant(root: Any, root_name: str) -> Tuple[Optional[Any], Optional[str]]:
    if root is None:
        return None, root_name

    queue = [(root, root_name)]
    seen = set()
    fallback = None
    while queue:
        obj, name = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if _looks_like_wan_model(obj):
            return obj, name
        if fallback is None and callable(getattr(obj, "forward_orig", None)) and hasattr(obj, "blocks"):
            fallback = (obj, name)

        for attr, child in _iter_wrapper_children(obj):
            queue.append((child, f"{name}.{attr}"))

    if fallback is not None:
        return fallback
    return None, root_name


def _locate_wan_inner_model(model) -> Tuple[Optional[Any], Optional[str]]:
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        diffusion_model = None
    if diffusion_model is not None:
        inner, inner_name = _locate_wan_like_descendant(diffusion_model, "diffusion_model")
        if inner is not None:
            return inner, inner_name

    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return _locate_wan_like_descendant(outer.diffusion_model, "model.diffusion_model")
    if hasattr(model, "diffusion_model"):
        return _locate_wan_like_descendant(model.diffusion_model, "diffusion_model")
    return None, None


def _is_wan_family_model(model) -> bool:
    inner, _inner_name = _locate_wan_inner_model(model)
    return inner is not None and hasattr(inner, "blocks")


def _looks_like_minimax_h3_model(inner: Any) -> bool:
    if inner is None:
        return False
    if any(hasattr(inner, attr) for attr in ("double_blocks", "single_blocks", "input_blocks", "middle_block", "output_blocks")):
        return False
    if _looks_like_wan_model(inner):
        return False
    required_attrs = (
        "blocks",
        "video_patch_proj",
        "audio_patch_proj",
        "condition_proj",
        "final_layer",
        "token_refiner",
        "sigma_shift_video",
        "sigma_shift_audio",
    )
    return callable(getattr(inner, "_forward", None)) and all(hasattr(inner, attr) for attr in required_attrs)


def _locate_minimax_h3_descendant(root: Any, root_name: str) -> Tuple[Optional[Any], Optional[str]]:
    if root is None:
        return None, None

    queue = [(root, root_name)]
    seen = set()
    while queue:
        obj, name = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if _looks_like_minimax_h3_model(obj):
            return obj, name
        for attr, child in _iter_wrapper_children(obj):
            queue.append((child, f"{name}.{attr}"))
    return None, None


def _locate_minimax_h3_inner_model(model) -> Tuple[Optional[Any], Optional[str]]:
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        diffusion_model = None
    if diffusion_model is not None:
        inner, inner_name = _locate_minimax_h3_descendant(diffusion_model, "diffusion_model")
        if inner is not None:
            return inner, inner_name

    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        inner, inner_name = _locate_minimax_h3_descendant(outer.diffusion_model, "model.diffusion_model")
        if inner is not None:
            return inner, inner_name
    if hasattr(model, "diffusion_model"):
        return _locate_minimax_h3_descendant(model.diffusion_model, "diffusion_model")
    return None, None


def _is_minimax_h3_model(model) -> bool:
    inner, _inner_name = _locate_minimax_h3_inner_model(model)
    return inner is not None


def _get_minimax_h3_block_count(model) -> Tuple[int, Optional[str]]:
    inner, inner_name = _locate_minimax_h3_inner_model(model)
    if inner is None:
        raise ValueError("This node expects a native ComfyUI MiniMax H3 MODEL with the packed-sequence transformer structure.")
    try:
        total = len(inner.blocks)
    except (TypeError, AttributeError) as exc:
        raise ValueError("The detected native MiniMax H3 model does not expose a valid blocks list.") from exc
    if total <= 0:
        raise ValueError("The detected native MiniMax H3 model has no transformer blocks.")
    return total, inner_name


def _get_flux_block_counts(model) -> Tuple[int, int]:
    diffusion_model = model.get_model_object("diffusion_model")
    return len(diffusion_model.double_blocks), len(diffusion_model.single_blocks)


def _get_wan_block_count(model) -> Tuple[int, Optional[str]]:
    inner, inner_name = _locate_wan_inner_model(model)
    if inner is None or not hasattr(inner, "blocks"):
        raise ValueError("This node expects a WAN-family MODEL whose diffusion model exposes a blocks list.")
    return len(inner.blocks), inner_name


def _map_indices_to_wan_blocks(requested_1based: Sequence[int], total_blocks: int) -> WanMappedBlocks:
    mapped_0based: List[int] = []
    for index_1based in requested_1based:
        if index_1based > total_blocks:
            raise ValueError(f"WAN block index {index_1based} is outside the model range 1..{total_blocks}.")
        mapped_0based.append(index_1based - 1)
    mapped_0based = _dedupe_preserve_order(mapped_0based)
    return WanMappedBlocks(
        requested_1based=tuple(_dedupe_preserve_order(requested_1based)),
        mapped_1based=tuple(index + 1 for index in mapped_0based),
        mapped_0based=tuple(mapped_0based),
        total=total_blocks,
    )


def _map_indices_to_minimax_h3_blocks(requested_1based: Sequence[int], total_blocks: int) -> MiniMaxH3MappedBlocks:
    mapped_0based: List[int] = []
    for index_1based in requested_1based:
        if index_1based > total_blocks:
            raise ValueError(f"MiniMax H3 block index {index_1based} is outside the model range 1..{total_blocks}.")
        mapped_0based.append(index_1based - 1)
    return MiniMaxH3MappedBlocks(
        requested_1based=tuple(_dedupe_preserve_order(requested_1based)),
        mapped_0based=tuple(_dedupe_preserve_order(mapped_0based)),
        total=total_blocks,
    )


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

    def _reset_sigma_state(self):
        self._first_sigma_abs = None
        self._last_sigma_abs = None

    def cleanup(self, **kwargs):
        self._reset_sigma_state()
        cleanup = getattr(self.existing_wrapper, "cleanup", None)
        if callable(cleanup):
            return cleanup(**kwargs)
        return None

    def _normalized_sigma(self, timestep, device: torch.device, transformer_options: Optional[Dict] = None) -> torch.Tensor:
        t = _as_float_tensor(timestep, device=device)
        if t is None or t.numel() == 0:
            return torch.tensor([1.0], device=device, dtype=torch.float32)

        transformer_options = transformer_options or {}
        scheduled_first_sigma = _max_abs_positive(transformer_options.get("sample_sigmas", None), device=device)
        if scheduled_first_sigma is not None:
            # Prefer the current sampler's sigma schedule over wrapper-local
            # state. This makes interrupted/restarted runs independent even
            # when the next run begins at a lower sigma than the interrupted one.
            self._first_sigma_abs = scheduled_first_sigma
            self._last_sigma_abs = float(t.abs().max().item())
            return (t.abs() / scheduled_first_sigma).clamp(0.0, 1.0)

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
        state["normalized_sigma"] = self._normalized_sigma(timestep, device=device, transformer_options=transformer_options)
        state["reference_latents"] = _detect_reference_latents(c, transformer_options)
        wan_context_img_len = _detect_wan_context_image_tokens(c, transformer_options)
        if wan_context_img_len > 0:
            state["wan_context_img_len"] = wan_context_img_len
        else:
            state.pop("wan_context_img_len", None)
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


class WanBlockReplacePatch(nn.Module):
    def __init__(self, config: SharedConfig, preserve_image_context_prefix: bool, existing_patch=None):
        super().__init__()
        self.config = config
        self.preserve_image_context_prefix = bool(preserve_image_context_prefix)
        self.existing_patch = existing_patch
        self._reported_reference_latents = False
        self._reported_image_context = False

    def _call_next(self, args: Dict, extra: Dict):
        # Diff-Aid mutates the WAN context first, then preserves any patch that already owned this block.
        if self.existing_patch is not None:
            return self.existing_patch(args, extra)
        return extra["original_block"](args)

    def _maybe_report_scope(self, transformer_options: Dict, text_start: int):
        state = transformer_options.get(STATE_KEY, {}) or {}
        if state.get("reference_latents", False) and not self._reported_reference_latents:
            print("[ComfyUI-DiffAid-Patches] WAN reference latents detected; Diff-Aid modulation remains context-token-only.")
            self._reported_reference_latents = True
        if text_start > 0 and not self._reported_image_context:
            print(f"[ComfyUI-DiffAid-Patches] WAN image-context prefix detected; preserving first {text_start} context tokens.")
            self._reported_image_context = True

    def _text_start(self, transformer_options: Dict, token_count: int) -> int:
        if not self.preserve_image_context_prefix:
            return 0
        state = transformer_options.get(STATE_KEY, {}) or {}
        try:
            text_start = int(state.get("wan_context_img_len", 0))
        except (TypeError, ValueError):
            text_start = 0
        return max(0, min(text_start, token_count))

    def forward(self, args: Dict, extra: Dict):
        transformer_options = args.get("transformer_options", {}) or {}
        key_name = "txt"
        context = args.get(key_name, None)
        if not torch.is_tensor(context):
            key_name = "context"
            context = args.get(key_name, None)
        if not torch.is_tensor(context) or context.ndim < 3 or context.shape[1] <= 0:
            return self._call_next(args, extra)

        token_count = int(context.shape[1])
        text_start = self._text_start(transformer_options, token_count)
        self._maybe_report_scope(transformer_options, text_start)
        if text_start >= token_count:
            return self._call_next(args, extra)

        text_tokens = context[:, text_start:, :]
        alpha = _compute_alpha(text_tokens, text_tokens.shape[1], self.config, transformer_options)
        text_mod = text_tokens + text_tokens * alpha

        new_args = dict(args)
        if text_start == 0:
            new_args[key_name] = text_mod
        else:
            new_args[key_name] = torch.cat((context[:, :text_start, :], text_mod), dim=1)
        return self._call_next(new_args, extra)


def _minimax_h3_language_ranges(mod_segments: Any, row_count: int) -> Tuple[Tuple[int, int], ...]:
    if mod_segments is None:
        raise RuntimeError(
            "Native MiniMax H3 block arguments are missing 'mod_segments'; "
            "this ComfyUI build is incompatible with tag-1-only Diff-Aid text scoping."
        )
    if isinstance(mod_segments, (str, bytes)):
        raise RuntimeError("Native MiniMax H3 'mod_segments' must be a sequence of (start, stop, modulation_index) entries.")
    try:
        segment_count = len(mod_segments)
    except TypeError as exc:
        raise RuntimeError("Native MiniMax H3 'mod_segments' must be an indexable sequence.") from exc

    language_ranges: List[Tuple[int, int]] = []
    previous_stop = 0
    for position in range(segment_count):
        try:
            segment = mod_segments[position]
        except (KeyError, TypeError, IndexError) as exc:
            raise RuntimeError("Native MiniMax H3 'mod_segments' must be an indexable sequence.") from exc
        if isinstance(segment, (str, bytes)):
            raise RuntimeError(f"Native MiniMax H3 mod_segments[{position}] is not a valid segment tuple.")
        try:
            segment_length = len(segment)
        except TypeError as exc:
            raise RuntimeError(f"Native MiniMax H3 mod_segments[{position}] is not a valid segment tuple.") from exc
        if segment_length < 3:
            raise RuntimeError(
                f"Native MiniMax H3 mod_segments[{position}] must contain start, stop, and modulation index."
            )
        try:
            start = int(segment[0])
            stop = int(segment[1])
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise RuntimeError(
                f"Native MiniMax H3 mod_segments[{position}] contains non-integer-compatible range metadata."
            ) from exc

        if start < 0 or stop < start or stop > row_count:
            raise RuntimeError(
                f"Native MiniMax H3 mod_segments[{position}] range [{start}, {stop}) "
                f"is outside packed row range 0..{row_count}."
            )
        if start < previous_stop:
            raise RuntimeError(
                f"Native MiniMax H3 mod_segments[{position}] range [{start}, {stop}) "
                "overlaps or descends relative to an earlier segment."
            )
        previous_stop = max(previous_stop, stop)
        if start == stop:
            continue

        modulation_metadata = segment[2]
        if torch.is_tensor(modulation_metadata) and modulation_metadata.numel() != 1:
            # Current native H3 denoise masks represent target video/audio
            # modulation rows as one LongTensor index per packed row. Text
            # presentation runs remain scalar tag-1 segments, so per-row target
            # metadata must be validated but excluded from Diff-Aid text scope.
            expected_rows = stop - start
            if modulation_metadata.ndim != 1 or int(modulation_metadata.numel()) != expected_rows:
                raise RuntimeError(
                    f"Native MiniMax H3 mod_segments[{position}] per-row modulation metadata "
                    f"must be a 1D tensor with {expected_rows} entries; got shape "
                    f"{tuple(modulation_metadata.shape)}."
                )
            if (
                modulation_metadata.dtype == torch.bool
                or torch.is_floating_point(modulation_metadata)
                or torch.is_complex(modulation_metadata)
            ):
                raise RuntimeError(
                    f"Native MiniMax H3 mod_segments[{position}] per-row modulation metadata "
                    f"must use an integer dtype; got {modulation_metadata.dtype}."
                )
            continue

        try:
            if torch.is_tensor(modulation_metadata):
                modulation_index = int(modulation_metadata.item())
            else:
                modulation_index = int(modulation_metadata)
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise RuntimeError(
                f"Native MiniMax H3 mod_segments[{position}] contains non-integer-compatible modulation metadata."
            ) from exc

        if modulation_index % 3 == 1:
            language_ranges.append((start, stop))
    return tuple(language_ranges)


class MiniMaxH3BlockReplacePatch(nn.Module):
    def __init__(self, config: SharedConfig, existing_patch=None):
        super().__init__()
        self.config = config
        self.existing_patch = existing_patch

    def _call_next(self, args: Dict, extra: Dict):
        if self.existing_patch is not None:
            return self.existing_patch(args, extra)
        return extra["original_block"](args)

    def forward(self, args: Dict, extra: Dict):
        img = args.get("img", None)
        if not torch.is_tensor(img) or img.ndim != 2:
            shape = getattr(img, "shape", None)
            raise RuntimeError(
                f"Native MiniMax H3 block replacement expected args['img'] as a 2D packed tensor; got shape {shape}."
            )

        language_ranges = _minimax_h3_language_ranges(args.get("mod_segments", None), int(img.shape[0]))
        if not language_ranges:
            return self._call_next(args, extra)

        total_text_rows = sum(stop - start for start, stop in language_ranges)
        transformer_options = args.get("transformer_options", {}) or {}
        reference = img.new_empty((1, total_text_rows, 1))
        alpha = _compute_alpha(reference, total_text_rows, self.config, transformer_options)[0]
        new_img = img.clone()
        alpha_offset = 0
        for start, stop in language_ranges:
            count = stop - start
            source = img[start:stop]
            segment_alpha = alpha[alpha_offset:alpha_offset + count]
            new_img[start:stop] = source + source * segment_alpha
            alpha_offset += count

        new_args = dict(args)
        new_args["img"] = new_img
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


def _get_existing_dit_replace_patch(model, block_kind: str, stage_index: int):
    # ComfyUI stores Flux and WAN block replacements under the same DiT namespace.
    transformer_options = model.model_options.get("transformer_options", {})
    patches_replace = transformer_options.get("patches_replace", {})
    dit_patches = patches_replace.get("dit", {})
    return dit_patches.get((block_kind, stage_index))


def _get_existing_flux_replace_patch(model, stage: str, stage_index: int):
    block_kind = "double_block" if stage == "double" else "single_block"
    return _get_existing_dit_replace_patch(model, block_kind, stage_index)


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


def _cache_bool(value: Any) -> bool:
    return bool(value)


def _cache_float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return default


def _cache_str(value: Any) -> str:
    if value is None:
        return ""
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized.lower()


def _cache_block_indices(value: Any) -> str:
    try:
        return ",".join(str(index) for index in _parse_combined_block_indices(str(value)))
    except (TypeError, ValueError):
        return _cache_str(value)


def _shared_config_fingerprint(
    *,
    enabled: bool,
    strength: float,
    sigma_start: float,
    sigma_end: float,
    sigma_ramp: float,
    token_weight_mode: str,
    token_tail: float,
    cond_only: bool,
) -> Tuple[Any, ...]:
    if not _cache_bool(enabled):
        return ("disabled",)
    return (
        "enabled",
        _cache_float(strength),
        _cache_float(sigma_start),
        _cache_float(sigma_end),
        _cache_float(sigma_ramp),
        _cache_str(token_weight_mode),
        _cache_float(token_tail),
        _cache_bool(cond_only),
    )


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

    @staticmethod
    def IS_CHANGED(
        model=None,
        enabled: bool = True,
        block_preset: str = "paper_sparse_flux_double_only_safe",
        block_indices: str = "1,15,36,41,48",
        strength: float = 0.5,
        sigma_start: float = 0.0,
        sigma_end: float = 1.0,
        sigma_ramp: float = 0.0,
        token_weight_mode: str = "none",
        token_tail: float = 0.35,
        apply_single_stream: bool = False,
        cond_only: bool = True,
    ):
        use_custom_indices = _cache_str(block_preset) == "custom_combined_indices"
        return (
            "flux_sparse",
            _cache_str(block_preset),
            _cache_str(block_indices) if use_custom_indices else "",
            _cache_bool(apply_single_stream),
            *_shared_config_fingerprint(
                enabled=enabled,
                strength=strength,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                sigma_ramp=sigma_ramp,
                token_weight_mode=token_weight_mode,
                token_tail=token_tail,
                cond_only=cond_only,
            ),
        )

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


class WanDiffAidSparsePatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "block_preset": (
                    ["paper_sparse_flux_remapped", "custom_block_indices"],
                    {"default": "paper_sparse_flux_remapped"},
                ),
                "block_indices": ("STRING", {"default": "1,15,36,41,48", "multiline": False, "advanced": True}),
                "strength": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
                "sigma_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_ramp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001, "advanced": True}),
                "token_weight_mode": (["none", "linear", "exponential"], {"default": "none"}),
                "token_tail": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "preserve_image_context_prefix": ("BOOLEAN", {"default": True, "advanced": True}),
                "cond_only": ("BOOLEAN", {"default": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "patch"
    CATEGORY = "model_patches/diffaid"

    @staticmethod
    def IS_CHANGED(
        model=None,
        enabled: bool = True,
        block_preset: str = "paper_sparse_flux_remapped",
        block_indices: str = "1,15,36,41,48",
        strength: float = 0.35,
        sigma_start: float = 0.0,
        sigma_end: float = 1.0,
        sigma_ramp: float = 0.0,
        token_weight_mode: str = "none",
        token_tail: float = 0.35,
        preserve_image_context_prefix: bool = True,
        cond_only: bool = True,
    ):
        use_custom_indices = _cache_str(block_preset) == "custom_block_indices"
        return (
            "wan_sparse",
            _cache_str(block_preset),
            _cache_str(block_indices) if use_custom_indices else "",
            _cache_bool(preserve_image_context_prefix),
            *_shared_config_fingerprint(
                enabled=enabled,
                strength=strength,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                sigma_ramp=sigma_ramp,
                token_weight_mode=token_weight_mode,
                token_tail=token_tail,
                cond_only=cond_only,
            ),
        )

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
        preserve_image_context_prefix: bool = True,
        cond_only: bool = True,
    ):
        if not enabled:
            return model, "disabled"
        if not _is_wan_family_model(model):
            raise ValueError("This node only supports WAN-family MODEL objects that expose a single blocks list. Use the Flux node for double_blocks/single_blocks models.")
        if sigma_start > sigma_end:
            raise ValueError(f"sigma_start ({sigma_start}) must be <= sigma_end ({sigma_end}).")

        total_blocks, inner_name = _get_wan_block_count(model)
        if block_preset == "paper_sparse_flux_remapped":
            requested_indices = list(PAPER_SPARSE_FLUX_COMBINED_BLOCKS)
            mapped_indices = _remap_paper_sparse_flux_to_single_block_list(total_blocks)
            preset_name = f"paper_sparse_flux_remapped_from_flux1(source_total={PAPER_SOURCE_FLUX_TOTAL_BLOCKS})"
        else:
            requested_indices = _parse_combined_block_indices(block_indices)
            mapped_indices = requested_indices
            preset_name = "custom_block_indices"

        mapped = _map_indices_to_wan_blocks(mapped_indices, total_blocks)
        config = SharedConfig(float(strength), float(sigma_start), float(sigma_end), float(sigma_ramp), token_weight_mode, float(token_tail), bool(cond_only))

        patched = model.clone()
        existing_model_wrapper = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(SharedTimestepWrapper(existing_wrapper=existing_model_wrapper))

        for block_index in mapped.mapped_0based:
            existing_patch = _get_existing_dit_replace_patch(patched, "double_block", block_index)
            patched.set_model_patch_replace(
                WanBlockReplacePatch(config, preserve_image_context_prefix=preserve_image_context_prefix, existing_patch=existing_patch),
                "dit",
                "double_block",
                block_index,
            )

        summary = (
            f"wan_sparse preset={preset_name}; requested_blocks=[{_fmt_ints(requested_indices)}]; "
            f"mapped_blocks=[{_fmt_ints(mapped.mapped_1based)}]; blocks_0based=[{_fmt_ints(mapped.mapped_0based)}]; "
            f"cond_only={str(config.cond_only).lower()}; preserve_image_context_prefix={str(bool(preserve_image_context_prefix)).lower()}; "
            f"strength={config.strength:.3f}; normalized_sigma_window=[{config.sigma_start:.3f}, {config.sigma_end:.3f}] ramp={config.sigma_ramp:.3f}; "
            f"token_weight_mode={config.token_weight_mode}; token_tail={config.token_tail:.3f}; reference_latents=runtime_detected_context_only; "
            f"model_total_blocks={mapped.total}; inner={inner_name or '-'}"
        )
        print(f"[ComfyUI-DiffAid-Patches] {summary}")
        return patched, summary


class MiniMaxH3DiffAidSparsePatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "block_indices": ("STRING", {"default": "1,13,25,37,50", "multiline": False}),
                "strength": ("FLOAT", {"default": 0.20, "min": -1.0, "max": 1.0, "step": 0.01}),
                "sigma_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_ramp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001, "advanced": True}),
                "token_weight_mode": (["none", "linear", "exponential"], {"default": "none"}),
                "token_tail": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "cond_only": ("BOOLEAN", {"default": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "patch"
    CATEGORY = "model_patches/diffaid"

    @staticmethod
    def IS_CHANGED(
        model=None,
        enabled: bool = True,
        block_indices: str = "1,13,25,37,50",
        strength: float = 0.20,
        sigma_start: float = 0.0,
        sigma_end: float = 1.0,
        sigma_ramp: float = 0.0,
        token_weight_mode: str = "none",
        token_tail: float = 0.35,
        cond_only: bool = True,
    ):
        return (
            "minimax_h3_sparse",
            _cache_bool(enabled),
            _cache_block_indices(block_indices),
            _cache_float(strength),
            _cache_float(sigma_start),
            _cache_float(sigma_end),
            _cache_float(sigma_ramp),
            _cache_str(token_weight_mode),
            _cache_float(token_tail),
            _cache_bool(cond_only),
        )

    def patch(
        self,
        model,
        enabled: bool,
        block_indices: str,
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
        if not _is_minimax_h3_model(model):
            raise ValueError(
                "This node only supports native ComfyUI MiniMax H3 MODEL objects with packed text/visual/audio/video rows."
            )
        if sigma_start > sigma_end:
            raise ValueError(f"sigma_start ({sigma_start}) must be <= sigma_end ({sigma_end}).")

        requested_indices = _parse_combined_block_indices(block_indices)
        total_blocks, inner_name = _get_minimax_h3_block_count(model)
        mapped = _map_indices_to_minimax_h3_blocks(requested_indices, total_blocks)
        config = SharedConfig(
            float(strength),
            float(sigma_start),
            float(sigma_end),
            float(sigma_ramp),
            token_weight_mode,
            float(token_tail),
            bool(cond_only),
        )

        patched = model.clone()
        existing_model_wrapper = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(SharedTimestepWrapper(existing_wrapper=existing_model_wrapper))

        for block_index in mapped.mapped_0based:
            existing_patch = _get_existing_dit_replace_patch(patched, "double_block", block_index)
            patched.set_model_patch_replace(
                MiniMaxH3BlockReplacePatch(config, existing_patch=existing_patch),
                "dit",
                "double_block",
                block_index,
            )

        summary = (
            f"minimax_h3_sparse requested_blocks=[{_fmt_ints(mapped.requested_1based)}]; "
            f"blocks_0based=[{_fmt_ints(mapped.mapped_0based)}]; cond_only={str(config.cond_only).lower()}; "
            f"strength={config.strength:.3f}; normalized_sigma_window=[{config.sigma_start:.3f}, {config.sigma_end:.3f}] ramp={config.sigma_ramp:.3f}; "
            f"token_weight_mode={config.token_weight_mode}; token_tail={config.token_tail:.3f}; "
            f"text_scope=native_mod_segments_tag_1_only; model_total_blocks={mapped.total}; inner={inner_name or '-'}"
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

    @staticmethod
    def IS_CHANGED(
        model=None,
        enabled: bool = True,
        stage_filter: str = "all",
        block_targets: str = "",
        strength: float = 0.35,
        sigma_start: float = 0.0,
        sigma_end: float = 1.0,
        sigma_ramp: float = 0.0,
        token_weight_mode: str = "linear",
        token_tail: float = 0.35,
        cond_only: bool = True,
    ):
        has_targets = bool(_cache_str(block_targets))
        return (
            "sdxl_cross_attention",
            _cache_str(stage_filter) if not has_targets else "",
            _cache_str(block_targets),
            *_shared_config_fingerprint(
                enabled=enabled,
                strength=strength,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                sigma_ramp=sigma_ramp,
                token_weight_mode=token_weight_mode,
                token_tail=token_tail,
                cond_only=cond_only,
            ),
        )

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
    "WanDiffAidSparsePatch": WanDiffAidSparsePatchNode,
    "MiniMaxH3DiffAidSparsePatch": MiniMaxH3DiffAidSparsePatchNode,
    "SDXLDiffAidCrossAttentionPatch": SDXLDiffAidCrossAttentionPatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2DiffAidSparsePatch": "Flux-family Diff-Aid Sparse Patch",
    "WanDiffAidSparsePatch": "WAN Diff-Aid Sparse Patch",
    "MiniMaxH3DiffAidSparsePatch": "MiniMax H3 Diff-Aid Sparse Patch",
    "SDXLDiffAidCrossAttentionPatch": "SDXL Diff-Aid Cross-Attention Patch",
}
