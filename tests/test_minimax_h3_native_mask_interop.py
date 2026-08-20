from __future__ import annotations

import torch

import nodes


def _config():
    return nodes.SharedConfig(
        strength=0.25,
        sigma_start=0.0,
        sigma_end=1.0,
        sigma_ramp=0.0,
        token_weight_mode="none",
        token_tail=0.35,
        cond_only=True,
    )


def test_masked_per_row_metadata_composes_with_existing_h3_block_replacement():
    img = torch.arange(1, 25, dtype=torch.float32).reshape(8, 3)
    mod_segments = [
        (0, 2, 1),
        (2, 4, torch.tensor([5, 8], dtype=torch.long)),
        (4, 8, torch.tensor([6, 9, 12, 15], dtype=torch.long)),
    ]
    calls = {"existing": 0, "original": 0}

    def existing(call_args, extra):
        calls["existing"] += 1
        assert torch.equal(call_args["img"][:2], img[:2] * 1.25)
        assert torch.equal(call_args["img"][2:], img[2:])
        return extra["original_block"](call_args)

    def original(call_args):
        calls["original"] += 1
        return {"img": call_args["img"], "owner": "original"}

    patch = nodes.MiniMaxH3BlockReplacePatch(_config(), existing_patch=existing)
    output = patch(
        {
            "img": img,
            "mod_segments": mod_segments,
            "transformer_options": {},
        },
        {"original_block": original},
    )

    assert calls == {"existing": 1, "original": 1}
    assert output["owner"] == "original"
    assert torch.equal(output["img"][:2], img[:2] * 1.25)
    assert torch.equal(output["img"][2:], img[2:])
