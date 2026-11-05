from __future__ import annotations

import copy

import pytest
import torch

import nodes


class FakeKrea2:
    def __init__(self, blocks=28, txtlayers=12):
        self.blocks = [object() for _ in range(blocks)]
        self.txtlayers = txtlayers
        self.txtfusion = object()
        self.txtmlp = object()
        self.pe_embedder = object()
        self.first = object()

    def process_img(self):
        pass

    def _unpack_context(self):
        pass


class FakeModel:
    def __init__(self, inner=None, options=None, counter=None):
        self.inner = inner or FakeKrea2()
        self.model_options = copy.deepcopy(options if options is not None else {"transformer_options": {}})
        self.counter = counter if counter is not None else {"clones": 0}

    def get_model_object(self, key):
        assert key == "diffusion_model"
        return self.inner

    def clone(self):
        self.counter["clones"] += 1
        return FakeModel(self.inner, self.model_options, self.counter)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def set_model_attn1_patch(self, patch):
        transformer_options = self.model_options.setdefault("transformer_options", {})
        transformer_options.setdefault("patches", {}).setdefault("attn1_patch", []).append(patch)


def config(**overrides):
    values = dict(
        strength=0.2,
        sigma_start=0.0,
        sigma_end=1.0,
        sigma_ramp=0.0,
        token_weight_mode="none",
        token_tail=0.35,
        cond_only=True,
    )
    values.update(overrides)
    return nodes.SharedConfig(**values)


def patch_model(model=None, **overrides):
    kwargs = dict(
        enabled=True,
        block_preset="uniform_five_experimental",
        block_indices="1,8,15,22,28",
        strength=0.2,
        attention_target="keys_and_values",
        sigma_start=0.0,
        sigma_end=1.0,
        sigma_ramp=0.0,
        token_weight_mode="none",
        token_tail=0.35,
        cond_only=True,
    )
    kwargs.update(overrides)
    return nodes.Krea2DiffAidSparsePatchNode().patch(model or FakeModel(), **kwargs)


def tensors(batch=1, length=6):
    q = torch.arange(1, batch * 4 * length * 2 + 1, dtype=torch.float32).reshape(batch, 4, length, 2)
    k = torch.arange(1, batch * 2 * length * 2 + 1, dtype=torch.float32).reshape(batch, 2, length, 2)
    v = k + 100.0
    return q, k, v


def test_model_identification_and_compiled_wrapper():
    assert nodes._get_krea2_block_count(FakeModel()) == 28
    wrapped = type("Compiled", (), {"_orig_mod": FakeKrea2()})()
    assert nodes._get_krea2_block_count(FakeModel(wrapped)) == 28


@pytest.mark.parametrize("inner", [
    type("Flux", (), {"double_blocks": [], "single_blocks": [], "txt_in": object()})(),
    type("Wan", (), {"blocks": [], "patch_embedding": object()})(),
    object(),
])
def test_rejects_other_architectures(inner):
    with pytest.raises(ValueError, match="native ComfyUI Krea 2"):
        nodes._get_krea2_block_count(FakeModel(inner))


def test_rejects_incompatible_text_fusion_and_no_blocks():
    with pytest.raises(ValueError, match="12-layer"):
        nodes._get_krea2_block_count(FakeModel(FakeKrea2(txtlayers=8)))
    with pytest.raises(ValueError, match="any transformer"):
        nodes._get_krea2_block_count(FakeModel(FakeKrea2(blocks=0)))


def test_passthrough_disabled_and_zero_strength_without_clone():
    model = FakeModel(inner=object())
    assert patch_model(model, enabled=False)[0] is model
    assert patch_model(model, strength=0.0)[0] is model
    assert model.counter["clones"] == 0


def test_preset_maps_and_installs_one_patch_without_mutating_input():
    model = FakeModel()
    patched, summary = patch_model(model)
    assert patched is not model
    assert model.model_options["transformer_options"] == {}
    assert model.counter["clones"] == 1
    p = patched.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    assert p.selected_blocks == frozenset({0, 7, 14, 21, 27})
    assert "mapped_blocks_1based=[1, 8, 15, 22, 28]" in summary
    assert isinstance(patched.model_options["model_function_wrapper"], nodes.SharedTimestepWrapper)


def test_preset_remaps_for_nonstandard_count_and_custom_indices():
    mapped, _ = patch_model(FakeModel(FakeKrea2(blocks=20)))
    p = mapped.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    assert p.selected_blocks == frozenset(i - 1 for i in nodes._remap_stage_indices((1, 8, 15, 22, 28), 28, 20))
    custom, _ = patch_model(block_preset="custom_block_indices", block_indices="2, 2, 9")
    p = custom.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    assert p.selected_blocks == frozenset({1, 8})


@pytest.mark.parametrize("changes,pattern", [
    ({"block_preset": "custom_block_indices", "block_indices": "29"}, "range 1..28"),
    ({"block_preset": "custom_block_indices", "block_indices": "0"}, "1-based"),
    ({"block_preset": "missing"}, "Unknown Krea 2 block preset"),
    ({"sigma_start": 0.7, "sigma_end": 0.3}, "sigma_start"),
    ({"attention_target": "bad"}, "attention target"),
])
def test_invalid_configuration_rejected(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        patch_model(**changes)


def test_attention_scales_only_selected_text_keys_and_values_and_preserves_inputs():
    p = nodes.Krea2AttentionPatch(config(), [3], "keys_and_values")
    q, k, v = tensors()
    original_k, original_v = k.clone(), v.clone()
    opts = {"block_type": "single", "block_index": 3, "img_slice": [2, 6], "normalized_sigma": 1.0}
    result = p(q, k, v, extra_options=opts)
    assert set(result) == {"k", "v"}
    torch.testing.assert_close(result["k"][:, :, :2], k[:, :, :2] * 1.2)
    torch.testing.assert_close(result["v"][:, :, :2], v[:, :, :2] * 1.2)
    torch.testing.assert_close(result["k"][:, :, 2:], k[:, :, 2:])
    torch.testing.assert_close(result["v"][:, :, 2:], v[:, :, 2:])
    torch.testing.assert_close(k, original_k)
    torch.testing.assert_close(v, original_v)
    assert q.shape[1] == 4 and result["k"].shape[1] == 2  # GQA
    assert result["k"].data_ptr() != k.data_ptr()


def test_attention_values_only_preserves_keys_and_query():
    p = nodes.Krea2AttentionPatch(config(), [0], "values_only")
    q, k, v = tensors()
    result = p(q, k, v, extra_options={"block_type": "single", "block_index": 0, "img_slice": [3, 6]})
    assert set(result) == {"v"}
    torch.testing.assert_close(result["v"][:, :, 3:], v[:, :, 3:])


@pytest.mark.parametrize("options", [
    {},
    {"block_type": "single", "block_index": 4, "img_slice": [2, 6]},
    {"block_type": "single", "block_index": 1, "img_slice": [2, 6]},
    {"block_type": "single", "block_index": 3, "img_slice": [0, 6]},
])
def test_text_fusion_unselected_blocks_and_empty_text_are_noops(options):
    q, k, v = tensors()
    assert nodes.Krea2AttentionPatch(config(), [3], "keys_and_values")(q, k, v, extra_options=options) == {}


def test_attention_conditional_only_with_two_batch_rows():
    p = nodes.Krea2AttentionPatch(config(cond_only=True), [0], "keys_and_values")
    q, k, v = tensors(batch=2)
    opts = {"block_type": "single", "block_index": 0, "img_slice": [3, 6], "cond_or_uncond": [0, 1]}
    result = p(q, k, v, extra_options=opts)
    torch.testing.assert_close(result["k"][0, :, :3], k[0, :, :3] * 1.2)
    torch.testing.assert_close(result["k"][1], k[1])
    torch.testing.assert_close(result["v"][1], v[1])


def test_sigma_window_and_token_decay_from_runtime_state():
    p = nodes.Krea2AttentionPatch(
        config(sigma_start=0.5, sigma_end=1.0, token_weight_mode="linear", token_tail=0.0), [0], "values_only",
    )
    q, k, v = tensors()
    opts = {
        "block_type": "single", "block_index": 0, "img_slice": [3, 6],
        nodes.STATE_KEY: {"normalized_sigma": torch.tensor([0.75])},
    }
    result = p(q, k, v, extra_options=opts)
    expected = torch.tensor([1.2, 1.1, 1.0])
    torch.testing.assert_close(result["v"][:, :, :3], v[:, :, :3] * expected[None, None, :, None])
    opts[nodes.STATE_KEY]["normalized_sigma"] = torch.tensor([0.25])
    result = p(q, k, v, extra_options=opts)
    torch.testing.assert_close(result["v"], v)


@pytest.mark.parametrize("image_slice,pattern", [
    (None, "img_slice"),
    ([-1, 6], "negative"),
    ([7, 6], "exceeds"),
])
def test_malformed_text_prefix_fails_closed(image_slice, pattern):
    q, k, v = tensors()
    with pytest.raises(ValueError, match=pattern):
        nodes.Krea2AttentionPatch(config(), [0], "keys_and_values")(
            q, k, v, extra_options={"block_type": "single", "block_index": 0, "img_slice": image_slice},
        )


def test_existing_attention_patch_and_model_wrapper_preserved():
    class Wrapper:
        def __call__(self, function, params):
            return function(params["input"], params["timestep"], **params["c"])

    old_patch = lambda q, k, v, **kwargs: {"k": k}
    existing_wrapper = Wrapper()
    model = FakeModel(options={
        "transformer_options": {"patches": {"attn1_patch": [old_patch]}},
        "model_function_wrapper": existing_wrapper,
    })
    patched, _ = patch_model(model)
    patches = patched.model_options["transformer_options"]["patches"]["attn1_patch"]
    assert patches[0] is old_patch
    assert isinstance(patches[1], nodes.Krea2AttentionPatch)
    assert isinstance(patched.model_options["model_function_wrapper"].existing_wrapper, Wrapper)


def test_cache_fingerprint_tracks_effective_changes_only():
    check = nodes.Krea2DiffAidSparsePatchNode.IS_CHANGED
    default = check()
    assert default == check(block_indices="2,9")
    assert default != check(attention_target="values_only")
    assert default != check(strength=0.2)
    assert check(block_preset="custom_block_indices", block_indices="1,2") == check(
        block_preset="custom_block_indices", block_indices="1, 2, 2",
    )
    assert check(block_preset="custom_block_indices", block_indices="1,2") != check(
        block_preset="custom_block_indices", block_indices="1,3",
    )
