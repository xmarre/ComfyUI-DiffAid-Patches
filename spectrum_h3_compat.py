from __future__ import annotations

import math
from typing import Any

try:
    from . import nodes as _nodes
except ImportError:  # pragma: no cover - direct test/import fallback
    import nodes as _nodes


EXTERNAL_PATCH_CONTRACTS_KEY = "spectrum_h3_external_patch_contracts"
EXTERNAL_PATCH_RUNTIME_KEY = "spectrum_h3_external_patch_runtime"
EXTERNAL_PATCH_SCHEMA_VERSION = 1
EXTERNAL_PATCH_PROVIDER = "comfyui-diffaid-patches"
EXTERNAL_PATCH_KIND = "text_activation_modulation"
EXTERNAL_PATCH_ARCHITECTURE = "minimax_h3"
EXTERNAL_PATCH_SCOPE = "native_mod_segments_tag_1_only"

_INSTANCE_ATTR = "_spectrum_h3_external_patch_instance_id"
_ORIGINAL_H3_PATCH = None
_ORIGINAL_INJECT_STATE = None
_INSTALLED = False


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
    return {
        "schema_version": EXTERNAL_PATCH_SCHEMA_VERSION,
        "provider": EXTERNAL_PATCH_PROVIDER,
        "kind": EXTERNAL_PATCH_KIND,
        "architecture": EXTERNAL_PATCH_ARCHITECTURE,
        "instance_id": instance_id,
        "block_indices_0based": list(mapped.mapped_0based),
        "model_block_count": int(mapped.total),
        "strength": float(config.strength),
        "sigma_start": float(config.sigma_start),
        "sigma_end": float(config.sigma_end),
        "sigma_ramp": float(config.sigma_ramp),
        "token_weight_mode": str(config.token_weight_mode),
        "token_tail": float(config.token_tail),
        "cond_only": bool(config.cond_only),
        "scope": EXTERNAL_PATCH_SCOPE,
    }


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
            return max(0.0, min(1.0, abs(last_value) / first_value))

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
    return max(0.0, min(1.0, value)) if math.isfinite(value) else None


def _inject_state_with_spectrum_contract(self, c: dict[str, Any], timestep: Any) -> dict[str, Any]:
    assert _ORIGINAL_INJECT_STATE is not None
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
    assert _ORIGINAL_H3_PATCH is not None
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
    contracts = _existing_contracts(model_options)
    instance_id = _next_instance_id(contracts)
    contracts.append(_descriptor(mapped, config, instance_id))
    model_options[EXTERNAL_PATCH_CONTRACTS_KEY] = tuple(contracts)
    patched.model_options = model_options

    wrapper = model_options.get("model_function_wrapper")
    if not isinstance(wrapper, _nodes.SharedTimestepWrapper):
        raise RuntimeError("MiniMax H3 Diff-Aid wrapper ownership changed before compatibility registration")
    setattr(wrapper, _INSTANCE_ATTR, instance_id)

    return patched, f"{summary}; spectrum_h3_contract=v1:{instance_id}"


def install_spectrum_h3_compat() -> None:
    global _INSTALLED, _ORIGINAL_H3_PATCH, _ORIGINAL_INJECT_STATE
    if _INSTALLED:
        return
    _ORIGINAL_H3_PATCH = _nodes.MiniMaxH3DiffAidSparsePatchNode.patch
    _ORIGINAL_INJECT_STATE = _nodes.SharedTimestepWrapper._inject_state
    _nodes.MiniMaxH3DiffAidSparsePatchNode.patch = _patch_h3_with_spectrum_contract
    _nodes.SharedTimestepWrapper._inject_state = _inject_state_with_spectrum_contract
    _INSTALLED = True


__all__ = [
    "EXTERNAL_PATCH_ARCHITECTURE",
    "EXTERNAL_PATCH_CONTRACTS_KEY",
    "EXTERNAL_PATCH_KIND",
    "EXTERNAL_PATCH_PROVIDER",
    "EXTERNAL_PATCH_RUNTIME_KEY",
    "EXTERNAL_PATCH_SCHEMA_VERSION",
    "EXTERNAL_PATCH_SCOPE",
    "install_spectrum_h3_compat",
]
