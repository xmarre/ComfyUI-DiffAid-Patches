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

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        wrappers = self.model_options.setdefault("wrappers", {})
        wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

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
        modulation_target="keys_and_values",
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
    ({"modulation_target": "bad"}, "modulation target"),
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
    assert default != check(modulation_target="values_only")
    assert default != check(strength=0.2)
    assert check(block_preset="custom_block_indices", block_indices="1,2") == check(
        block_preset="custom_block_indices", block_indices="1, 2, 2",
    )
    assert check(block_preset="custom_block_indices", block_indices="1,2") != check(
        block_preset="custom_block_indices", block_indices="1,3",
    )


def _krea2_attention_like_upstream(q, k, v, patches, extra_options):
    """Mirror of ComfyUI comfy/ldm/krea2/model.py Attention.forward patch dispatch.

    attn1_patch runs after QK normalization and before RoPE / GQA key-value
    repetition. Text RoPE positions are all zero (identity rotation), so RoPE
    is omitted here; it does not change the text-prefix result.
    """
    if "block_index" in extra_options:
        for p in patches:
            out = p(q, k, v, pe=None, attn_mask=None, extra_options=extra_options.copy())
            q, k, v = out.get("q", q), out.get("k", k), out.get("v", v)
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    probs = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
    return probs @ v, probs


def test_upstream_style_gqa_dispatch_values_only_and_keys_and_values():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 7, 8)
    k = torch.randn(1, 2, 7, 8)
    v = torch.randn(1, 2, 7, 8)
    opts = {"block_type": "single", "block_index": 5, "img_slice": [3, 7], "normalized_sigma": 1.0}
    reference, ref_probs = _krea2_attention_like_upstream(q, k, v, [], opts)

    values_only = nodes.Krea2AttentionPatch(config(strength=0.2), [5], "values_only")
    out, probs = _krea2_attention_like_upstream(q, k, v, [values_only], opts)
    torch.testing.assert_close(probs, ref_probs)
    v_text = v[:, :, :3].repeat_interleave(2, dim=1)
    torch.testing.assert_close(out - reference, ref_probs[..., :3] @ (0.2 * v_text))

    both = nodes.Krea2AttentionPatch(config(strength=0.2), [5], "keys_and_values")
    _, probs = _krea2_attention_like_upstream(q, k, v, [both], opts)
    logits = q @ k.repeat_interleave(2, dim=1).transpose(-1, -2) / 8 ** 0.5
    logits[..., :3] = logits[..., :3] * 1.2
    torch.testing.assert_close(probs, torch.softmax(logits, dim=-1))

    # Text-fusion attention runs before the DiT loop sets block_index.
    fusion_opts = {k_: v_ for k_, v_ in opts.items() if k_ not in ("block_index", "block_type")}
    out, _ = _krea2_attention_like_upstream(q, k, v, [both], fusion_opts)
    torch.testing.assert_close(out, reference)


class _RecordingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
        self.inputs.append(x.clone())
        return x + 1.0


class _FakeKrea2DiT(torch.nn.Module):
    """Mirrors the native Krea 2 block loop's call convention and options."""

    def __init__(self, blocks=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_RecordingBlock() for _ in range(blocks))

    def run(self, combined, txtlen, transformer_options):
        transformer_options = dict(transformer_options)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txtlen, combined.shape[1]]
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            combined = block(combined, None, None, None, timestep_zero_index=None,
                             transformer_options=transformer_options)
        return combined


class _Executor:
    def __init__(self, dit):
        self.class_obj = dit

    def __call__(self, *args, **kwargs):
        return self.class_obj.run(*args, **kwargs)


def test_default_target_installs_text_residual_wrapper_not_attention_patch():
    patched, summary = nodes.Krea2DiffAidSparsePatchNode().patch(
        FakeModel(), True, "uniform_five_experimental", "", 0.15, "text_residual",
        0.0, 1.0, 0.0, "none", 0.35, True,
    )
    assert "attn1_patch" not in patched.model_options["transformer_options"].get("patches", {})
    wrappers = patched.model_options["wrappers"]["diffusion_model"][nodes.Krea2TextResidualWrapper.WRAPPER_KEY]
    assert len(wrappers) == 1 and wrappers[0].selected_blocks == (0, 7, 14, 21, 27)
    assert "modulation_target=text_residual" in summary
    defaults = nodes.Krea2DiffAidSparsePatchNode.INPUT_TYPES()["required"]["modulation_target"]
    assert defaults[1]["default"] == "text_residual"


def test_text_residual_scales_only_selected_block_text_prefix_and_removes_hooks():
    dit = _FakeKrea2DiT()
    wrapper = nodes.Krea2TextResidualWrapper(config(strength=0.5), [1])
    x = torch.randn(2, 5, 3)
    opts = {"cond_or_uncond": [0, 1], "normalized_sigma": 1.0}
    out = wrapper(_Executor(dit), x, 2, opts)

    torch.testing.assert_close(dit.blocks[0].inputs[0], x)
    into_selected = dit.blocks[1].inputs[0]
    expected = x + 1.0
    torch.testing.assert_close(into_selected[0, :2], expected[0, :2] * 1.5)  # cond row, text
    torch.testing.assert_close(into_selected[0, 2:], expected[0, 2:])  # cond row, image
    torch.testing.assert_close(into_selected[1], expected[1])  # uncond row
    assert all(len(b._forward_pre_hooks) == 0 for b in dit.blocks)

    reference = _FakeKrea2DiT().run(x, 2, opts)
    torch.testing.assert_close(out[:, 2:], reference[:, 2:])
    torch.testing.assert_close(out[1], reference[1])


def test_text_residual_hooks_removed_on_error_and_bounds_checked():
    dit = _FakeKrea2DiT()
    wrapper = nodes.Krea2TextResidualWrapper(config(), [0])

    class Failing(_Executor):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        wrapper(Failing(dit), torch.zeros(1, 3, 2), 1, {})
    assert all(len(b._forward_pre_hooks) == 0 for b in dit.blocks)
    assert wrapper._active is False
    with pytest.raises(ValueError, match="exceed"):
        nodes.Krea2TextResidualWrapper(config(), [4])(_Executor(dit), torch.zeros(1, 3, 2), 1, {})


def test_text_residual_zero_text_prefix_and_sigma_window_are_identity():
    dit = _FakeKrea2DiT()
    x = torch.randn(1, 4, 3)
    out = nodes.Krea2TextResidualWrapper(config(), [0, 2])(_Executor(dit), x, 0, {})
    torch.testing.assert_close(out, _FakeKrea2DiT().run(x, 0, {}))
    windowed = nodes.Krea2TextResidualWrapper(config(sigma_start=0.0, sigma_end=0.5), [0, 2])
    out = windowed(_Executor(_FakeKrea2DiT()), x, 2, {"normalized_sigma": 0.9})
    torch.testing.assert_close(out, _FakeKrea2DiT().run(x, 2, {}))
