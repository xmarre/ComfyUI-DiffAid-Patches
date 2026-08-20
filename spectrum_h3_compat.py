from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import struct
import sys
from typing import Any

import torch


def _load_sibling_nodes():
    sibling = Path(__file__).resolve().with_name("nodes.py")
    existing = sys.modules.get("nodes")
    existing_file = getattr(existing, "__file__", None)
    if existing_file is not None and Path(existing_file).resolve() == sibling:
        return existing

    module_name = "_comfyui_diffaid_patches_nodes"
    cached = sys.modules.get(module_name)
    cached_file = getattr(cached, "__file__", None)
    if cached_file is not None and Path(cached_file).resolve() == sibling:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, sibling)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Diff-Aid sibling module from {sibling}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


if __package__:
    from . import nodes as _nodes
else:  # pragma: no cover - direct test/import fallback
    _nodes = _load_sibling_nodes()


EXTERNAL_PATCH_CONTRACTS_KEY = "spectrum_h3_external_patch_contracts"
EXTERNAL_PATCH_RUNTIME_KEY = "spectrum_h3_external_patch_runtime"
EXTERNAL_PATCH_SCHEMA_VERSION = 1
EXTERNAL_PATCH_PROVIDER = "comfyui-diffaid-patches"
EXTERNAL_PATCH_KIND = "text_activation_modulation"
EXTERNAL_PATCH_ARCHITECTURE = "minimax_h3"
EXTERNAL_PATCH_SCOPE = "native_mod_segments_tag_1_only"
H3_REFINEMENT_REQUEST_KEY = "h3_refinement"
H3_REFINEMENT_API = 1

_INSTANCE_ATTR = "_spectrum_h3_external_patch_instance_id"
_INSTALL_MARKER_ATTR = "_spectrum_h3_compat_install_marker"
_INSTALL_MARKER_VALUE = f"{EXTERNAL_PATCH_PROVIDER}:v{EXTERNAL_PATCH_SCHEMA_VERSION}"
_ORIGINAL_H3_PATCH = None
_ORIGINAL_INJECT_STATE = None
_ORIGINAL_NORMALIZED_SIGMA = None
_INSTALLED = False


def _float32_scalar(value: float) -> float:
    """Round a Python scalar exactly once to IEEE-754 binary32."""
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def _existing_contracts(model_options: dict[str, Any]) -> list[Any]:
    value = model_options.get(EXTERNAL_PATCH_CONTRACTS_KEY)
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return list(value)
    # Preserve foreign/malformed metadata rather than silently replacing it.
    # Spectrum will validate the combined declaration and fail safe if needed.
    return [value]


def _next_instance_id(existing: list[Any]) -> str:
    used = {
        str(value.get("instance_id"))
        for value in existing
        if isinstance(value, dict) and value.get("instance_id") is not None
    }
    ordinal = 1
    while f"diffaid-h3-{ordinal}" in used:
        ordinal += 1
    return f"diffaid-h3-{ordinal}"


def _descriptor(mapped: Any, config: Any, instance_id: str) -> dict[str, Any]:
    sigma_ramp = float(config.sigma_ramp)
    # Diff-Aid compares hard-window bounds against a float32 normalized-sigma
    # tensor, so those Python widget scalars are effectively cast to binary32.
    # Publish those effective hard boundaries so Spectrum's Python-side regime
    # comparison classifies the same inclusive window.
    if sigma_ramp == 0.0:
        sigma_start = _float32_scalar(config.sigma_start)
        sigma_end = _float32_scalar(config.sigma_end)
    else:
        sigma_start = float(config.sigma_start)
        sigma_end = float(config.sigma_end)

    return {
        "schema_version": EXTERNAL_PATCH_SCHEMA_VERSION,
        "provider": EXTERNAL_PATCH_PROVIDER,
        "kind": EXTERNAL_PATCH_KIND,
        "architecture": EXTERNAL_PATCH_ARCHITECTURE,
        "instance_id": instance_id,
        "block_indices_0based": list(mapped.mapped_0based),
        "model_block_count": int(mapped.total),
        "strength": float(config.strength),
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "sigma_ramp": sigma_ramp,
        "token_weight_mode": str(config.token_weight_mode),
        "token_tail": float(config.token_tail),
        "cond_only": bool(config.cond_only),
        "scope": EXTERNAL_PATCH_SCOPE,
    }


def _refinement_sigma_reference(transformer_options: dict[str, Any] | None) -> float | None:
    """Resolve the explicit full-trajectory coordinate for sampler-2 refinement.

    Ordinary Diff-Aid runs intentionally normalize against the first sigma of the
    current invocation.  A learned-latent refinement is different: it resumes at
    low sigma, so renormalizing its first call to 1.0 creates an artificial hard
    off->on patch transition.  Only a complete API-v1 refinement contract opts in
    to the full H3 sigma reference.
    """
    if not isinstance(transformer_options, dict):
        return None
    request = transformer_options.get(H3_REFINEMENT_REQUEST_KEY)
    if not isinstance(request, dict):
        return None

    api = request.get("api")
    active = request.get("active")
    prefix = request.get("min_actual_prefix_steps")
    sigma_reference = request.get("sigma_reference")
    if type(api) is not int or type(active) is not bool or type(prefix) is not int:
        return None
    if api != H3_REFINEMENT_API or active is not True or prefix < 0:
        return None
    if isinstance(sigma_reference, bool) or not isinstance(sigma_reference, (int, float)):
        return None
    sigma_reference = float(sigma_reference)
    if not math.isfinite(sigma_reference) or sigma_reference <= 0.0:
        return None
    return sigma_reference


def _normalized_sigma_with_refinement(
    self,
    timestep: Any,
    device: torch.device,
    transformer_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    reference = _refinement_sigma_reference(transformer_options)
    if reference is None:
        if _ORIGINAL_NORMALIZED_SIGMA is None:
            raise RuntimeError("Spectrum H3 compatibility lost the original SharedTimestepWrapper._normalized_sigma target")
        return _ORIGINAL_NORMALIZED_SIGMA(
            self,
            timestep,
            device=device,
            transformer_options=transformer_options,
        )

    t = _nodes._as_float_tensor(timestep, device=device)
    if t is None or t.numel() == 0:
        return torch.tensor([1.0], device=device, dtype=torch.float32)
    self._first_sigma_abs = reference
    self._last_sigma_abs = float(t.abs().max().item())
    return (t.abs() / reference).clamp(0.0, 1.0)


def _normalized_sigma_scalar(wrapper: Any, injected: dict[str, Any]) -> float | None:
    first = getattr(wrapper, "_first_sigma_abs", None)
    last = getattr(wrapper, "_last_sigma_abs", None)
    if first is not None and last is not None:
        try:
            first_value = float(first)
            last_value = float(last)
        except (TypeError, ValueError):
            first_value = 0.0
            last_value = math.nan
        if math.isfinite(first_value) and math.isfinite(last_value) and first_value > 0.0:
            # SharedTimestepWrapper computes the coordinate in a float32 tensor.
            # These operands have already been materialized as Python scalars, so
            # round the reconstructed ratio back to binary32 without another GPU
            # scalar read/synchronization before publishing it to Spectrum.
            normalized = max(0.0, min(1.0, abs(last_value) / first_value))
            return _float32_scalar(normalized)

    # The normal path above reuses the Python scalars already materialized by
    # SharedTimestepWrapper._normalized_sigma, so it adds no CUDA synchronization.
    # A CPU-only fallback keeps synthetic/direct calls diagnosable without forcing
    # a device scalar transfer in the exceptional GPU case.
    state = (injected.get("transformer_options") or {}).get(_nodes.STATE_KEY, {}) or {}
    normalized = state.get("normalized_sigma")
    if getattr(normalized, "device", None) is not None and normalized.device.type != "cpu":
        return None
    try:
        value = float(normalized.reshape(-1)[0].item()) if hasattr(normalized, "reshape") else float(normalized)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return _float32_scalar(max(0.0, min(1.0, value)))


def _inject_state_with_spectrum_contract(self, c: dict[str, Any], timestep: Any) -> dict[str, Any]:
    if _ORIGINAL_INJECT_STATE is None:
        raise RuntimeError("Spectrum H3 compatibility lost the original SharedTimestepWrapper._inject_state target")
    injected = _ORIGINAL_INJECT_STATE(self, c, timestep)
    instance_id = getattr(self, _INSTANCE_ATTR, None)
    if not instance_id:
        return injected

    result = dict(injected)
    transformer_options = dict(result.get("transformer_options", {}) or {})
    existing_runtime = transformer_options.get(EXTERNAL_PATCH_RUNTIME_KEY)
    if existing_runtime is None:
        runtime_entries: list[Any] = []
    elif isinstance(existing_runtime, (tuple, list)):
        runtime_entries = list(existing_runtime)
    else:
        runtime_entries = [existing_runtime]
    runtime_entries.append(
        {
            "schema_version": EXTERNAL_PATCH_SCHEMA_VERSION,
            "provider": EXTERNAL_PATCH_PROVIDER,
            "instance_id": str(instance_id),
            "normalized_sigma": _normalized_sigma_scalar(self, result),
        }
    )
    transformer_options[EXTERNAL_PATCH_RUNTIME_KEY] = tuple(runtime_entries)
    result["transformer_options"] = transformer_options
    return result


def _patch_h3_with_spectrum_contract(
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
    if _ORIGINAL_H3_PATCH is None:
        raise RuntimeError("Spectrum H3 compatibility lost the original MiniMaxH3DiffAidSparsePatchNode.patch target")
    patched, summary = _ORIGINAL_H3_PATCH(
        self,
        model,
        enabled,
        block_indices,
        strength,
        sigma_start,
        sigma_end,
        sigma_ramp,
        token_weight_mode,
        token_tail,
        cond_only,
    )
    if not enabled or float(strength) == 0.0:
        return patched, summary

    # Resolve the descriptor through the same helpers used by the actual H3 patch.
    # This records the effective 0-based block set and detected topology rather
    # than the raw user string.
    requested = _nodes._parse_combined_block_indices(block_indices)
    total_blocks, _ = _nodes._get_minimax_h3_block_count(model)
    mapped = _nodes._map_indices_to_minimax_h3_blocks(requested, total_blocks)
    config = _nodes.SharedConfig(
        float(strength),
        float(sigma_start),
        float(sigma_end),
        float(sigma_ramp),
        token_weight_mode,
        float(token_tail),
        bool(cond_only),
    )

    model_options = dict(getattr(patched, "model_options", {}) or {})
    wrapper = model_options.get("model_function_wrapper")
    if not isinstance(wrapper, _nodes.SharedTimestepWrapper):
        raise RuntimeError("MiniMax H3 Diff-Aid wrapper ownership changed before compatibility registration")

    contracts = _existing_contracts(model_options)
    instance_id = _next_instance_id(contracts)
    contracts.append(_descriptor(mapped, config, instance_id))
    model_options[EXTERNAL_PATCH_CONTRACTS_KEY] = tuple(contracts)
    patched.model_options = model_options
    setattr(wrapper, _INSTANCE_ATTR, instance_id)

    return patched, f"{summary}; spectrum_h3_contract=v1:{instance_id}"


def _is_installed_replacement(value: Any) -> bool:
    return getattr(value, _INSTALL_MARKER_ATTR, None) == _INSTALL_MARKER_VALUE


def install_spectrum_h3_compat() -> None:
    global _INSTALLED, _ORIGINAL_H3_PATCH, _ORIGINAL_INJECT_STATE, _ORIGINAL_NORMALIZED_SIGMA

    current_patch = _nodes.MiniMaxH3DiffAidSparsePatchNode.patch
    current_inject = _nodes.SharedTimestepWrapper._inject_state
    current_normalized = _nodes.SharedTimestepWrapper._normalized_sigma
    patch_installed = _is_installed_replacement(current_patch)
    inject_installed = _is_installed_replacement(current_inject)
    normalized_installed = _is_installed_replacement(current_normalized)
    if patch_installed or inject_installed or normalized_installed:
        if not (patch_installed and inject_installed and normalized_installed):
            raise RuntimeError("Spectrum H3 compatibility is only partially installed")
        _INSTALLED = True
        return

    _ORIGINAL_H3_PATCH = current_patch
    _ORIGINAL_INJECT_STATE = current_inject
    _ORIGINAL_NORMALIZED_SIGMA = current_normalized
    setattr(_patch_h3_with_spectrum_contract, _INSTALL_MARKER_ATTR, _INSTALL_MARKER_VALUE)
    setattr(_inject_state_with_spectrum_contract, _INSTALL_MARKER_ATTR, _INSTALL_MARKER_VALUE)
    setattr(_normalized_sigma_with_refinement, _INSTALL_MARKER_ATTR, _INSTALL_MARKER_VALUE)
    _nodes.MiniMaxH3DiffAidSparsePatchNode.patch = _patch_h3_with_spectrum_contract
    _nodes.SharedTimestepWrapper._inject_state = _inject_state_with_spectrum_contract
    _nodes.SharedTimestepWrapper._normalized_sigma = _normalized_sigma_with_refinement
    _INSTALLED = True


__all__ = [
    "EXTERNAL_PATCH_ARCHITECTURE",
    "EXTERNAL_PATCH_CONTRACTS_KEY",
    "EXTERNAL_PATCH_KIND",
    "EXTERNAL_PATCH_PROVIDER",
    "EXTERNAL_PATCH_RUNTIME_KEY",
    "EXTERNAL_PATCH_SCHEMA_VERSION",
    "EXTERNAL_PATCH_SCOPE",
    "H3_REFINEMENT_API",
    "H3_REFINEMENT_REQUEST_KEY",
    "install_spectrum_h3_compat",
]
