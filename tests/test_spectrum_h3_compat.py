from __future__ import annotations

import copy

import pytest
import torch

import nodes
import spectrum_h3_compat as compat


compat.install_spectrum_h3_compat()


class FakeH3Inner:
    def __init__(self, block_count=50):
        self.blocks = [object() for _ in range(block_count)]
        self.video_patch_proj = object()
        self.audio_patch_proj = object()
        self.condition_proj = object()
        self.final_layer = object()
        self.token_refiner = object()
        self.sigma_shift_video = 12.0
        self.sigma_shift_audio = 3.0

    def _forward(self):
        return None


class FakeModelPatcher:
    def __init__(self, inner, model_options=None, clone_counter=None):
        self.inner = inner
        self.model_options = copy.deepcopy(model_options or {})
        self.clone_counter = clone_counter if clone_counter is not None else {"count": 0}

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.inner

    def clone(self):
        self.clone_counter["count"] += 1
        return FakeModelPatcher(self.inner, self.model_options, self.clone_counter)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def set_model_patch_replace(self, patch, namespace, block_kind, block_index):
        transformer_options = self.model_options.setdefault("transformer_options", {})
        patches_replace = transformer_options.setdefault("patches_replace", {})
        patches_replace.setdefault(namespace, {})[(block_kind, block_index)] = patch


def kwargs(**overrides):
    values = {
        "enabled": True,
        "block_indices": "1,13,25,37,50",
        "strength": 0.2,
        "sigma_start": 0.0,
        "sigma_end": 1.0,
        "sigma_ramp": 0.0,
        "token_weight_mode": "none",
        "token_tail": 0.35,
        "cond_only": True,
    }
    values.update(overrides)
    return values


def _patch(model=None, **overrides):
    model = model or FakeModelPatcher(FakeH3Inner())
    return nodes.MiniMaxH3DiffAidSparsePatchNode().patch(model, **kwargs(**overrides))


def _contracts(model):
    return model.model_options.get(compat.EXTERNAL_PATCH_CONTRACTS_KEY, ())


def _invoke_wrapper(wrapper, *, timestep=500.0, sample_sigmas=(1000.0, 500.0)):
    seen = {}

    def model_function(input_x, timestep, **conditioning):
        seen.update(conditioning)
        return input_x

    params = {
        "input": torch.ones(1),
        "timestep": torch.tensor([timestep]),
        "c": {
            "transformer_options": {
                "sample_sigmas": torch.tensor(sample_sigmas, dtype=torch.float32)
            }
        },
    }
    output = wrapper(model_function, params)
    return output, seen


def test_enabled_nonzero_h3_publishes_resolved_versioned_descriptor():
    patched, summary = _patch()
    descriptors = _contracts(patched)

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor == {
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "kind": "text_activation_modulation",
        "architecture": "minimax_h3",
        "instance_id": "diffaid-h3-1",
        "block_indices_0based": [0, 12, 24, 36, 49],
        "model_block_count": 50,
        "strength": 0.2,
        "sigma_start": 0.0,
        "sigma_end": 1.0,
        "sigma_ramp": 0.0,
        "token_weight_mode": "none",
        "token_tail": 0.35,
        "cond_only": True,
        "scope": "native_mod_segments_tag_1_only",
    }
    assert descriptor["block_indices_0based"][-1] == descriptor["model_block_count"] - 1
    assert "spectrum_h3_contract=v1:diffaid-h3-1" in summary


def test_descriptor_tracks_every_behaviorally_relevant_runtime_setting():
    patched, _ = _patch(
        block_indices="2,9,50",
        strength=-0.33,
        sigma_start=0.2,
        sigma_end=0.75,
        sigma_ramp=0.08,
        token_weight_mode="linear",
        token_tail=0.6,
        cond_only=False,
    )
    descriptor = _contracts(patched)[0]
    assert descriptor["block_indices_0based"] == [1, 8, 49]
    assert descriptor["strength"] == -0.33
    assert descriptor["sigma_start"] == 0.2
    assert descriptor["sigma_end"] == 0.75
    assert descriptor["sigma_ramp"] == 0.08
    assert descriptor["token_weight_mode"] == "linear"
    assert descriptor["token_tail"] == 0.6
    assert descriptor["cond_only"] is False


def test_contract_is_owned_only_by_clone_and_source_nested_options_are_unchanged():
    source_options = {
        "transformer_options": {"unrelated": {"value": 7}},
        "unrelated_top": [1, 2, 3],
    }
    model = FakeModelPatcher(FakeH3Inner(), source_options)
    before = copy.deepcopy(model.model_options)
    patched, _ = _patch(model)

    assert model.model_options == before
    assert compat.EXTERNAL_PATCH_CONTRACTS_KEY not in model.model_options
    assert _contracts(patched)
    assert patched.model_options is not model.model_options
    assert patched.model_options["transformer_options"] is not model.model_options["transformer_options"]


def test_disabled_and_zero_strength_do_not_advertise_active_modulation():
    model = FakeModelPatcher(FakeH3Inner())
    disabled, _ = _patch(model, enabled=False)
    zero, _ = _patch(model, strength=0.0)

    assert disabled is model
    assert not _contracts(disabled)
    assert not _contracts(zero)
    zero_wrapper = zero.model_options["model_function_wrapper"]
    _, seen = _invoke_wrapper(zero_wrapper)
    assert compat.EXTERNAL_PATCH_RUNTIME_KEY not in seen["transformer_options"]


def test_stacked_h3_nodes_preserve_distinct_descriptors_and_runtime_entries():
    first, _ = _patch(strength=0.2)
    second, _ = _patch(first, strength=0.2)
    descriptors = _contracts(second)

    assert [value["instance_id"] for value in descriptors] == [
        "diffaid-h3-1",
        "diffaid-h3-2",
    ]
    assert len(descriptors) == 2
    wrapper = second.model_options["model_function_wrapper"]
    _, seen = _invoke_wrapper(wrapper)
    runtime_entries = seen["transformer_options"][compat.EXTERNAL_PATCH_RUNTIME_KEY]
    assert [value["instance_id"] for value in runtime_entries] == [
        "diffaid-h3-1",
        "diffaid-h3-2",
    ]
    assert [value["normalized_sigma"] for value in runtime_entries] == pytest.approx([0.5, 0.5])


def test_runtime_state_reuses_exact_shared_timestep_normalization_without_stale_generation_state():
    patched, _ = _patch(sigma_start=0.55, sigma_end=1.0, sigma_ramp=0.0)
    wrapper = patched.model_options["model_function_wrapper"]

    _, first = _invoke_wrapper(wrapper, timestep=550.0, sample_sigmas=(1000.0, 550.0))
    first_entry = first["transformer_options"][compat.EXTERNAL_PATCH_RUNTIME_KEY][0]
    assert first_entry["normalized_sigma"] == pytest.approx(0.55)
    assert wrapper._first_sigma_abs is None
    assert wrapper._last_sigma_abs is None

    # A fresh invocation uses its own sample_sigmas normalization. No previous
    # generation/reference sigma survives SharedTimestepWrapper's finally cleanup.
    _, second = _invoke_wrapper(wrapper, timestep=200.0, sample_sigmas=(800.0, 200.0))
    second_entry = second["transformer_options"][compat.EXTERNAL_PATCH_RUNTIME_KEY][0]
    assert second_entry["normalized_sigma"] == pytest.approx(0.25)
    assert wrapper._first_sigma_abs is None
    assert wrapper._last_sigma_abs is None


def test_wrapper_chain_still_invokes_preexisting_wrapper_once():
    calls = []

    def existing_wrapper(model_function, params):
        calls.append("existing")
        return model_function(params["input"], params["timestep"], **params["c"])

    model = FakeModelPatcher(
        FakeH3Inner(),
        {"model_function_wrapper": existing_wrapper},
    )
    patched, _ = _patch(model)
    wrapper = patched.model_options["model_function_wrapper"]
    _invoke_wrapper(wrapper)
    assert calls == ["existing"]


def test_final_block_replacement_chain_still_receives_diffaid_rows_once():
    calls = {"existing": 0, "original": 0}
    img = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    segments = [(0, 2, 1), (2, 4, 5)]

    def existing(args, extra):
        calls["existing"] += 1
        assert torch.equal(args["img"][:2], img[:2] * 1.2)
        return {"img": args["img"], "owner": "spectrum_capture"}

    patch = nodes.MiniMaxH3BlockReplacePatch(
        nodes.SharedConfig(0.2, 0.0, 1.0, 0.0, "none", 0.35, True),
        existing_patch=existing,
    )

    def original(args):
        calls["original"] += 1
        return {"img": args["img"]}

    output = patch(
        {"img": img, "mod_segments": segments, "transformer_options": {}},
        {"original_block": original},
    )
    assert output["owner"] == "spectrum_capture"
    assert calls == {"existing": 1, "original": 0}
