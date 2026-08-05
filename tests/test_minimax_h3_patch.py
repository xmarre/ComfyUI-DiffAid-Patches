from __future__ import annotations

import pytest
import torch

import nodes


def clone_options(value):
    if isinstance(value, dict):
        return {clone_options(key): clone_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_options(item) for item in value]
    return value


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
        self.model_options = clone_options(model_options or {})
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


def config(**overrides):
    values = {
        "strength": 0.2,
        "sigma_start": 0.0,
        "sigma_end": 1.0,
        "sigma_ramp": 0.0,
        "token_weight_mode": "none",
        "token_tail": 0.35,
        "cond_only": True,
    }
    values.update(overrides)
    return nodes.SharedConfig(**values)


def packed_case(dtype=torch.float32):
    img = torch.arange(1, 43, dtype=dtype).reshape(14, 3)
    mod_segments = [
        (0, 2, 1),   # linguistic text
        (2, 3, 0),   # vision pad in the presentation span
        (3, 5, 4),   # linguistic text at another timestep row
        (5, 7, 2),   # reference audio
        (7, 9, 3),   # reference image/video
        (9, 11, 5),  # target audio
        (11, 14, 6), # target video
    ]
    return img, mod_segments


def run_patch(patch, img, mod_segments, transformer_options=None):
    args = {
        "img": img,
        "mod_segments": mod_segments,
        "transformer_options": transformer_options or {},
        "unrelated": object(),
    }
    calls = {"original": 0, "received_args": None}

    def original_block(call_args):
        calls["original"] += 1
        calls["received_args"] = call_args
        return {"img": call_args["img"], "owner": "original"}

    output = patch(args, {"original_block": original_block})
    return output, args, calls


def test_model_detection_accepts_native_and_wrapped_h3():
    inner = FakeH3Inner()
    direct = FakeModelPatcher(inner)
    wrapped = FakeModelPatcher(type("Compiled", (), {"_orig_mod": inner})())

    assert nodes._is_minimax_h3_model(direct)
    assert nodes._locate_minimax_h3_inner_model(wrapped) == (inner, "diffusion_model._orig_mod")


@pytest.mark.parametrize(
    "inner",
    [
        type("Wan", (), {"blocks": [], "patch_embedding": object(), "head": object(), "forward_orig": lambda self: None})(),
        type("Flux", (), {"double_blocks": [], "single_blocks": [], "txt_in": object(), "forward_orig": lambda self: None})(),
        type("Sdxl", (), {"input_blocks": [], "middle_block": object(), "output_blocks": []})(),
        type("Generic", (), {"blocks": []})(),
    ],
)
def test_model_detection_rejects_other_families(inner):
    assert not nodes._is_minimax_h3_model(FakeModelPatcher(inner))


def test_index_parsing_mapping_and_current_default():
    requested = nodes._parse_combined_block_indices("1, 13, 13;25 37|50")
    mapped = nodes._map_indices_to_minimax_h3_blocks(requested, 50)

    assert requested == [1, 13, 25, 37, 50]
    assert mapped.mapped_0based == (0, 12, 24, 36, 49)


@pytest.mark.parametrize("text", ["0", "-1", "1,-2"])
def test_nonpositive_indices_fail(text):
    with pytest.raises(ValueError, match="1-based"):
        nodes._parse_combined_block_indices(text)


def test_index_above_detected_block_count_fails():
    with pytest.raises(ValueError, match=r"range 1\.\.49"):
        nodes._map_indices_to_minimax_h3_blocks([50], 49)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_row_scope_preserves_every_nonlinguistic_row(dtype):
    img, mod_segments = packed_case(dtype=dtype)
    patch = nodes.MiniMaxH3BlockReplacePatch(config(strength=0.25))
    output, _, calls = run_patch(patch, img, mod_segments)
    result = output["img"]

    linguistic = torch.tensor([0, 1, 3, 4])
    preserved = torch.tensor([2, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert torch.equal(result[linguistic], img[linguistic] * 1.25)
    assert torch.equal(result[preserved], img[preserved])
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert result.device == img.device
    assert calls["original"] == 1


def test_non_2d_packed_tensor_is_rejected():
    patch = nodes.MiniMaxH3BlockReplacePatch(config())
    with pytest.raises(RuntimeError, match="2D packed tensor"):
        run_patch(patch, torch.ones(1, 4, 3), [(0, 4, 1)])


def test_interrupted_text_runs_share_continuous_token_weight_order():
    img, mod_segments = packed_case()
    patch = nodes.MiniMaxH3BlockReplacePatch(
        config(strength=1.0, token_weight_mode="linear", token_tail=0.25)
    )
    output, _, _ = run_patch(patch, img, mod_segments)
    result = output["img"]
    expected_weights = torch.tensor([1.0, 0.75, 0.50, 0.25])

    assert torch.allclose(result[[0, 1, 3, 4]], img[[0, 1, 3, 4]] * (1.0 + expected_weights[:, None]))
    assert torch.equal(result[2], img[2])


@pytest.mark.parametrize("strength", [0.0, 0.3, -0.3])
def test_strength_behavior(strength):
    img, mod_segments = packed_case()
    patch = nodes.MiniMaxH3BlockReplacePatch(config(strength=strength))
    output, _, _ = run_patch(patch, img, mod_segments)

    assert torch.allclose(output["img"][[0, 1, 3, 4]], img[[0, 1, 3, 4]] * (1.0 + strength))


def test_sigma_window_inside_outside_and_ramp():
    img, mod_segments = packed_case()
    hard = nodes.MiniMaxH3BlockReplacePatch(config(strength=0.4, sigma_start=0.4, sigma_end=0.6))
    inside, _, _ = run_patch(hard, img, mod_segments, {"normalized_sigma": 0.5})
    outside, _, _ = run_patch(hard, img, mod_segments, {"normalized_sigma": 0.2})
    ramped = nodes.MiniMaxH3BlockReplacePatch(
        config(strength=0.4, sigma_start=0.5, sigma_end=0.6, sigma_ramp=0.2)
    )
    shoulder, _, _ = run_patch(ramped, img, mod_segments, {"normalized_sigma": 0.4})

    assert torch.allclose(inside["img"][0], img[0] * 1.4)
    assert torch.equal(outside["img"], img)
    assert torch.allclose(shoulder["img"][0], img[0] * 1.2)


@pytest.mark.parametrize(
    "cond_only,branch,expected_scale",
    [(True, [0], 1.2), (True, [1], 1.0), (False, [1], 1.2)],
)
def test_conditional_branch_filtering(cond_only, branch, expected_scale):
    img, mod_segments = packed_case()
    patch = nodes.MiniMaxH3BlockReplacePatch(config(cond_only=cond_only))
    output, _, _ = run_patch(patch, img, mod_segments, {"cond_or_uncond": branch})

    assert torch.allclose(output["img"][0], img[0] * expected_scale)


def test_existing_replacement_receives_modified_args_once_and_owns_result():
    img, mod_segments = packed_case()
    calls = {"existing": 0, "original": 0}
    sentinel = {"img": torch.tensor([123.0]), "owner": "spectrum_capture"}

    def existing(call_args, extra):
        calls["existing"] += 1
        assert torch.allclose(call_args["img"][0], img[0] * 1.2)
        assert call_args["img"] is not img
        return sentinel

    def original(_args):
        calls["original"] += 1
        return {"img": img}

    patch = nodes.MiniMaxH3BlockReplacePatch(config(), existing_patch=existing)
    output = patch(
        {"img": img, "mod_segments": mod_segments, "transformer_options": {}},
        {"original_block": original},
    )

    assert output is sentinel
    assert calls == {"existing": 1, "original": 0}


def test_original_block_is_called_once_without_existing_replacement():
    img, mod_segments = packed_case()
    patch = nodes.MiniMaxH3BlockReplacePatch(config())
    _, _, calls = run_patch(patch, img, mod_segments)
    assert calls["original"] == 1


def test_patch_does_not_mutate_args_tensor_or_unrelated_values():
    img, mod_segments = packed_case()
    patch = nodes.MiniMaxH3BlockReplacePatch(config())
    original_img = img.clone()
    output, args, calls = run_patch(patch, img, mod_segments)

    assert args["img"] is img
    assert torch.equal(img, original_img)
    assert output["img"] is not img
    assert calls["received_args"]["unrelated"] is args["unrelated"]


@pytest.mark.parametrize(
    "segments,match",
    [
        (None, "missing 'mod_segments'"),
        ([(0, 1)], "must contain"),
        ([("bad", 1, 1)], "non-integer-compatible"),
        ([(-1, 1, 1)], "outside packed row range"),
        ([(0, 15, 1)], "outside packed row range"),
        ([(0, 3, 1), (2, 4, 0)], "overlaps or descends"),
        ([(3, 4, 1), (1, 2, 0)], "overlaps or descends"),
    ],
)
def test_metadata_failures_are_descriptive(segments, match):
    img, _ = packed_case()
    patch = nodes.MiniMaxH3BlockReplacePatch(config())
    with pytest.raises(RuntimeError, match=match):
        run_patch(patch, img, segments)


def test_no_tag_one_rows_is_safe_noop():
    img, _ = packed_case()
    segments = [(0, 7, 0), (7, 14, 2)]
    received = []

    def original(args):
        received.append(args)
        return {"img": args["img"]}

    args = {"img": img, "mod_segments": segments, "transformer_options": {}}
    patch = nodes.MiniMaxH3BlockReplacePatch(config())
    output = patch(args, {"original_block": original})

    assert received == [args]
    assert output["img"] is img


def node_kwargs(**overrides):
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


def test_node_disabled_returns_same_model_without_clone():
    model = FakeModelPatcher(FakeH3Inner())
    output, summary = nodes.MiniMaxH3DiffAidSparsePatchNode().patch(model, **node_kwargs(enabled=False))

    assert output is model
    assert summary == "disabled"
    assert model.clone_counter["count"] == 0


def test_node_clones_once_preserves_original_options_and_composes_hooks():
    wrapper_calls = []

    def existing_wrapper(model_function, params):
        wrapper_calls.append(params)
        return model_function(params["input"], params["timestep"], **params["c"])

    existing_block = lambda args, extra: extra["original_block"](args)
    model_options = {
        "model_function_wrapper": existing_wrapper,
        "transformer_options": {"patches_replace": {"dit": {("double_block", 49): existing_block}}},
    }
    model = FakeModelPatcher(FakeH3Inner(), model_options)
    original_wrapper = model.model_options["model_function_wrapper"]
    original_dit = dict(model.model_options["transformer_options"]["patches_replace"]["dit"])
    patched, summary = nodes.MiniMaxH3DiffAidSparsePatchNode().patch(model, **node_kwargs())

    assert model.clone_counter["count"] == 1
    assert model.model_options["model_function_wrapper"] is original_wrapper
    assert model.model_options["transformer_options"]["patches_replace"]["dit"] == original_dit
    wrapper = patched.model_options["model_function_wrapper"]
    assert isinstance(wrapper, nodes.SharedTimestepWrapper)
    assert wrapper.existing_wrapper is original_wrapper
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert set(index for kind, index in dit if kind == "double_block") == {0, 12, 24, 36, 49}
    assert dit[("double_block", 49)].existing_patch is existing_block
    assert "text_scope=native_mod_segments_tag_1_only" in summary
    assert "model_total_blocks=50" in summary

    seen = {}

    def model_function(input_x, timestep, **conditioning):
        seen.update(conditioning)
        return input_x

    params = {
        "input": torch.ones(1),
        "timestep": torch.tensor([500.0]),
        "c": {"transformer_options": {"sample_sigmas": torch.tensor([1000.0, 500.0])}},
    }
    assert wrapper(model_function, params) is params["input"]
    assert wrapper_calls == [params]
    assert nodes.STATE_KEY in seen["transformer_options"]


def test_node_rejects_wrong_model_sigma_order_and_out_of_range_index():
    node = nodes.MiniMaxH3DiffAidSparsePatchNode()
    with pytest.raises(ValueError, match="only supports native"):
        node.patch(FakeModelPatcher(object()), **node_kwargs())
    with pytest.raises(ValueError, match="must be <="):
        node.patch(FakeModelPatcher(FakeH3Inner()), **node_kwargs(sigma_start=0.8, sigma_end=0.2))
    with pytest.raises(ValueError, match=r"range 1\.\.49"):
        node.patch(FakeModelPatcher(FakeH3Inner(49)), **node_kwargs())


def test_is_changed_normalizes_indices_and_varies_for_every_effective_setting():
    node = nodes.MiniMaxH3DiffAidSparsePatchNode
    base = node.IS_CHANGED(**node_kwargs())
    assert base == node.IS_CHANGED(**node_kwargs(block_indices="1, 13,13,25,37,50"))

    variants = [
        node_kwargs(enabled=False),
        node_kwargs(block_indices="1,2"),
        node_kwargs(strength=0.21),
        node_kwargs(sigma_start=0.1),
        node_kwargs(sigma_end=0.9),
        node_kwargs(sigma_ramp=0.1),
        node_kwargs(token_weight_mode="linear"),
        node_kwargs(token_tail=0.4),
        node_kwargs(cond_only=False),
    ]
    assert all(node.IS_CHANGED(**variant) != base for variant in variants)
    assert node.IS_CHANGED(**node_kwargs(enabled=False, strength=0.2)) != node.IS_CHANGED(
        **node_kwargs(enabled=False, strength=0.3)
    )


def test_registration_preserves_existing_nodes_and_adds_h3():
    expected = {
        "Flux2DiffAidSparsePatch",
        "WanDiffAidSparsePatch",
        "MiniMaxH3DiffAidSparsePatch",
        "SDXLDiffAidCrossAttentionPatch",
    }
    assert expected <= nodes.NODE_CLASS_MAPPINGS.keys()
    assert expected <= nodes.NODE_DISPLAY_NAME_MAPPINGS.keys()
    assert nodes.NODE_CLASS_MAPPINGS["MiniMaxH3DiffAidSparsePatch"] is nodes.MiniMaxH3DiffAidSparsePatchNode
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3DiffAidSparsePatch"] == "MiniMax H3 Diff-Aid Sparse Patch"
