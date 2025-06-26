import os
import sys

sys.path.append("/home/kaiwenguo/dev/rnd-ditwo/src")

from datalib import InputPath, OutputPath
from safetensors.torch import load_file, save_file
import torch


from ditwo.utils.pretrained import from_wan_dit_state_dict


import numpy as np
import re
import torch


def from_wan_dit_state_dict(state_dict: dict[str, torch.Tensor], t2v_model: bool = False) -> dict[str, torch.Tensor]:
    new_state_dict = {}

    weight = state_dict["patch_embedding.weight"]
    if t2v_model:
        new_shape = list(weight.shape)
        new_shape[1] = 36  # Extend the number of channels to 36
        new_weight = torch.zeros(new_shape, dtype=weight.dtype, device=weight.device)
        new_weight[:, : weight.size(1), ...] = weight
        weight = new_weight

    weight = weight.moveaxis(-4, -1).flatten(1)
    new_state_dict["patch_embedder.fc.weight"] = weight
    new_state_dict["patch_embedder.fc.bias"] = state_dict["patch_embedding.bias"]

    new_state_dict["text_embedder.0.weight"] = state_dict["text_embedding.0.weight"]
    new_state_dict["text_embedder.0.bias"] = state_dict["text_embedding.0.bias"]
    new_state_dict["text_embedder.2.weight"] = state_dict["text_embedding.2.weight"]
    new_state_dict["text_embedder.2.bias"] = state_dict["text_embedding.2.bias"]

    new_state_dict["time_embedder.mlp.fc1.weight"] = state_dict["time_embedding.0.weight"]
    new_state_dict["time_embedder.mlp.fc1.bias"] = state_dict["time_embedding.0.bias"]
    new_state_dict["time_embedder.mlp.fc2.weight"] = state_dict["time_embedding.2.weight"]
    new_state_dict["time_embedder.mlp.fc2.bias"] = state_dict["time_embedding.2.bias"]

    weight, bias = state_dict["time_projection.1.weight"], state_dict["time_projection.1.bias"]
    new_state_dict["time_projection.1.weight"] = weight
    new_state_dict["time_projection.1.bias"] = bias
    # weight, bias = weight.reshape(6, weight.size(0) // 6, -1), bias.reshape(6, weight.size(0) // 6)
    # new_state_dict["time_projection.1.weight"] = weight
    # new_state_dict["time_projection.1.bias"] = bias

    num_transformer_blocks = len(
        set(
            match.group(1)
            for match in (re.match(r"blocks\.(\d+)", key) for key in state_dict.keys())
            if match is not None
        )
    )

    for i in range(num_transformer_blocks):
        weight = state_dict[f"blocks.{i}.self_attn.norm_q.weight"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.q_norm.weight"] = weight * np.sqrt(weight.size(-1))
        weight = state_dict[f"blocks.{i}.self_attn.norm_k.weight"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.k_norm.weight"] = weight * np.sqrt(weight.size(-1))
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_q.weight"] = state_dict[f"blocks.{i}.self_attn.q.weight"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_q.bias"] = state_dict[f"blocks.{i}.self_attn.q.bias"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_k.weight"] = state_dict[f"blocks.{i}.self_attn.k.weight"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_k.bias"] = state_dict[f"blocks.{i}.self_attn.k.bias"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_v.weight"] = state_dict[f"blocks.{i}.self_attn.v.weight"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_v.bias"] = state_dict[f"blocks.{i}.self_attn.v.bias"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_out.weight"] = state_dict[f"blocks.{i}.self_attn.o.weight"]
        new_state_dict[f"transformer_blocks.{i}.self_attn.fc_out.bias"] = state_dict[f"blocks.{i}.self_attn.o.bias"]
        weight = state_dict[f"blocks.{i}.cross_attn.norm_q.weight"]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.q_norm.weight"] = weight * np.sqrt(weight.size(-1))
        weight = state_dict[f"blocks.{i}.cross_attn.norm_k.weight"]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.k_norm.weight"] = weight * np.sqrt(weight.size(-1))
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_q.weight"] = (
            state_dict[f"blocks.{i}.cross_attn.q.weight"] * state_dict[f"blocks.{i}.norm3.weight"][None, :]
        )
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_q.bias"] = (
            state_dict[f"blocks.{i}.cross_attn.q.bias"]
            + state_dict[f"blocks.{i}.cross_attn.q.weight"] @ state_dict[f"blocks.{i}.norm3.bias"]
        )
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_k.weight"] = state_dict[f"blocks.{i}.cross_attn.k.weight"]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_k.bias"] = state_dict[f"blocks.{i}.cross_attn.k.bias"]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_v.weight"] = state_dict[f"blocks.{i}.cross_attn.v.weight"]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_v.bias"] = state_dict[f"blocks.{i}.cross_attn.v.bias"]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_out.weight"] = state_dict[
            f"blocks.{i}.cross_attn.o.weight"
        ]
        new_state_dict[f"transformer_blocks.{i}.cross_attn.fc_out.bias"] = state_dict[f"blocks.{i}.cross_attn.o.bias"]

        new_state_dict[f"transformer_blocks.{i}.ffn.0.weight"] = state_dict[f"blocks.{i}.ffn.0.weight"]
        new_state_dict[f"transformer_blocks.{i}.ffn.0.bias"] = state_dict[f"blocks.{i}.ffn.0.bias"]
        new_state_dict[f"transformer_blocks.{i}.ffn.2.weight"] = state_dict[f"blocks.{i}.ffn.2.weight"]
        new_state_dict[f"transformer_blocks.{i}.ffn.2.bias"] = state_dict[f"blocks.{i}.ffn.2.bias"]
        new_state_dict[f"transformer_blocks.{i}.vision_time_modulation_bias"] = state_dict[f"blocks.{i}.modulation"]

        # reference time modulation bias initialized to zeros
        new_state_dict[f"transformer_blocks.{i}.reference_time_modulation_bias"] = torch.zeros_like(
            state_dict[f"blocks.{i}.modulation"],
            dtype=state_dict[f"blocks.{i}.modulation"].dtype,
            device=state_dict[f"blocks.{i}.modulation"].device,
        )

    new_state_dict["final_layer.fc.weight"] = state_dict["head.head.weight"]
    new_state_dict["final_layer.fc.bias"] = (
        state_dict["head.head.bias"] + state_dict["head.head.weight"] @ state_dict["head.modulation"][0, 0]
    )
    new_state_dict["final_layer.time_modulation_bias"] = state_dict["head.modulation"][:, 1:]

    return new_state_dict


def main():
    model_type = "wan_fan"
    if model_type == "wan_fan":
        checkpoint_path = InputPath(
            "/home/kaiwenguo/dev/models/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"
        ).resolve()
        state_dict = load_file(checkpoint_path)
        new_state_dict = from_wan_dit_state_dict(state_dict)
        new_checkpoint_path = OutputPath(
            "s3://synthesia-rnd-prd-third-party-models/wan/dit-ref-modulation/Wan2.1-I2V-1.3B-multires-bfloat16.safetensors"
        )
        save_file(new_state_dict, new_checkpoint_path.resolve())
        new_checkpoint_path.commit()
    elif model_type == "wan":
        checkpoint_path = InputPath("s3://synthesia-rnd-prd-third-party-models/wan/Wan2.1-I2V-14B-480P/").resolve()
        state_dicts = {}
        for checkpoint_path in checkpoint_path.glob("*.safetensors"):
            state_dicts.update(load_file(checkpoint_path))
        new_state_dict = from_wan_dit_state_dict(state_dicts)
        new_checkpoint_path = OutputPath(
            "s3://synthesia-rnd-prd-third-party-models/wan/dit-ref-modulation/Wan2.1-I2V-14B-480P-bfloat16.safetensors"
        )
        save_file(new_state_dict, new_checkpoint_path.resolve())
        new_checkpoint_path.commit()
    else:
        raise ValueError(f"Invalid model type: {model_type}")


if __name__ == "__main__":
    main()
