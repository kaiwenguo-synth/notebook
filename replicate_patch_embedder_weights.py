from __future__ import annotations

import sys

sys.path.append("/home/kaiwenguo/dev/rnd-ditwo-develop-hg/src")

import torch

from datalib import InputPath, OutputPath
from safetensors.torch import save_file

from ditwo.utils.inference_pipeline import load_tensors_from_file


def load_from_wan_weights(pretrained_dit_state_dict_path: InputPath) -> dict[str, torch.Tensor]:
    """
    Loads weights from original WAN checkpoint format and initializes patch embedder.

    The patch embedder weights are derived from the patch embedder by replicating the
    weights of the last in_features channels from the original patch embedder weights.

    Args:
        pretrained_dit_state_dict_path: Path to the WAN checkpoint
    """
    # Load the state dictionary
    resolved_path = pretrained_dit_state_dict_path.resolve()
    print(f"Loading WAN weights from resolved path: {resolved_path}")
    state_dict = load_tensors_from_file(resolved_path, remove_prefix="module.")

    # Initialize patch embedder from original patch embedder weights
    _initialize_patch_embedder(state_dict)

    return state_dict


def _initialize_patch_embedder(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Initializes patch embedder weights from original patch embedder weights.

    The patch embedder weight matrix has shape [out_features, total_input_features] where
    total_input_features = temporal_patch_size * patch_size^2 * input_channels.

    We replicate the last in_features channels of the original patch embedder weight to
    initialize the patch embedder.

    Args:
        state_dict: Model state dictionary to modify in-place
    """
    # Extract and reshape patch embedder weights
    patch_embedder_weight = _reshape_patch_embedder_weight(state_dict["patch_embedder.fc.weight"])

    # Replicate the last in_features channels of the patch embedder weight to initialize the patch embedder.
    # Note: Using hardcoded value for downsampling factor since config is not available
    temporal_downsampling_factor = 4  # Typical value for VAE models
    in_features = (patch_embedder_weight.shape[-1] - temporal_downsampling_factor) // 2
    new_patch_embedder_weight = torch.cat([patch_embedder_weight, patch_embedder_weight[..., -in_features:]], dim=-1)
    new_patch_embedder_weight = new_patch_embedder_weight.flatten(
        1, -1
    ).clone()  # [hidden_size, flattened_patch_features]

    # Initialize patch embedder bias from patch embedder bias
    patch_embedder_bias = state_dict["patch_embedder.fc.bias"].clone()

    # Update state dictionary
    state_dict["patch_embedder.fc.weight"] = new_patch_embedder_weight
    state_dict["patch_embedder.fc.bias"] = patch_embedder_bias

    print(
        f"Initialized patch embedder from patch embedder: "
        f"weight shape {new_patch_embedder_weight.shape}, bias shape {patch_embedder_bias.shape}"
    )


def _reshape_patch_embedder_weight(patch_weight: torch.Tensor) -> torch.Tensor:
    """
    Reshapes patch embedder weight from flattened to structured format.

    Transforms from [out_features, total_input_features] to
    [out_features, temporal_patch_size, patch_size, patch_size, input_channels_per_patch].

    Args:
        patch_weight: Original patch embedder weight tensor

    Returns:
        Reshaped weight tensor with explicit spatial and temporal dimensions
    """
    # Using typical DIT model config values since config is not available
    hidden_size = patch_weight.shape[0]
    patch_size_temporal = 1  # Typical value
    patch_size = 2  # Typical value

    reshaped_weight = patch_weight.reshape(
        hidden_size,
        patch_size_temporal,
        patch_size,
        patch_size,
        -1,  # Infer input channels per patch
    )

    print(f"Reshaped patch embedder weight: {patch_weight.shape} -> {reshaped_weight.shape}")

    return reshaped_weight


def main():
    pretrained_dit_state_dict_path = InputPath(
        # "s3://synthesia-rnd-prd-third-party-models/wan/dit/Wan2.1-I2V-14B-720P-bfloat16-notext.safetensors"
        "s3://synthesia-rnd-prd-third-party-models/wan/dit/Wan2.1-I2V-1.3B-multires-bfloat16-notext.safetensors"
    )

    state_dict = load_from_wan_weights(pretrained_dit_state_dict_path)

    modified_dit_state_dict_path = OutputPath(
        # "s3://synthesia-rnd-prd-third-party-models/wan/dit/Wan2.1-I2V-14B-720P-bfloat16-notext-extended-patch-embedder.safetensors"
        "s3://synthesia-rnd-prd-third-party-models/wan/dit/Wan2.1-I2V-1.3B-multires-bfloat16-notext-extended-patch-embedder.safetensors"
    )

    # Save the modified state dict to safetensors
    save_file(state_dict, modified_dit_state_dict_path.resolve())
    modified_dit_state_dict_path.commit()

    print(f"Modified state dict saved to: {modified_dit_state_dict_path}")


if __name__ == "__main__":
    main()
