import sys

# Disable torch.compile for this script
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Alternative approach - set compile backend to eager (uncomment if needed)
# import torch
# torch.set_default_device = lambda device: None  # Prevent early torch import issues
# torch._dynamo.config.suppress_errors = True
# torch.compile = lambda fn, **kwargs: fn  # Replace torch.compile with identity function

sys.path.append("/home/kaiwenguo/dev/rnd-ditwo-develop-hg/src")

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from datalib import InputPath, OutputPath

from ditwo.data.dataset import bbox_centers_from_skeletons, get_crop_or_pad_coordinates
from ditwo.models.vae.wan import WanVAE
from ditwo.configs.config_dataclasses import CropType, SkeletonRegion, WanVAEConfig
from ditwo.utils.ffmpeg import save_video
from ditwo.utils.misc import default_dtype, Size
from ditwo.utils.skeleton.cpu_renderer import DWPoseCPURenderer
from ditwo.utils.skeleton.io import load_safetensor_skeletons
from ditwo.utils.skeleton.transforms import box_crop_skeletons, center_crop_skeletons, resize_skeletons


def visualize_pca_components(
    control_latents,
    output_prefix,
    save_individual_frames=False,
    output_dir=".",
    h_upsample=8,
    w_upsample=8,
    t_upsample=4,
):
    """
    Perform PCA on control latents and visualize first 3 components as RGB.

    Args:
        control_latents: torch.Tensor of shape (B=1, T_l, W_l, H_l, C_l)
        output_prefix: str, prefix for output files (will be converted to OutputPath)
        save_individual_frames: bool, whether to save individual frame images
        output_dir: str, directory to write output files to
        h_upsample: int, height upsampling factor for PCA video (default: 8)
        w_upsample: int, width upsampling factor for PCA video (default: 8)
        t_upsample: int, temporal upsampling factor for PCA video (default: 4)
    """
    print(f"Control latents shape: {control_latents.shape}")

    # Remove batch dimension and move to CPU
    latents = control_latents.squeeze(0).float().cpu().numpy()  # (T_l, W_l, H_l, C_l)
    T_l, W_l, H_l, C_l = latents.shape

    # Reshape to (T_l * W_l * H_l, C_l) for PCA
    latents_flat = latents.reshape(-1, C_l)
    print(f"Flattened latents shape for PCA: {latents_flat.shape}")

    # Perform PCA
    pca = PCA(n_components=min(3, C_l))
    pca_components = pca.fit_transform(latents_flat)  # (T_l * W_l * H_l, 3)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Normalize each component to [0, 1] for RGB visualization
    pca_normalized = np.zeros_like(pca_components)
    for i in range(pca_components.shape[1]):
        comp = pca_components[:, i]
        comp_min, comp_max = comp.min(), comp.max()
        pca_normalized[:, i] = (comp - comp_min) / (comp_max - comp_min)

    # Reshape back to spatial dimensions
    pca_spatial = pca_normalized.reshape(T_l, W_l, H_l, 3)  # (T_l, W_l, H_l, 3)

    # Create base output directory OutputPath
    output_base = OutputPath(output_dir)

    if save_individual_frames:
        # Save individual frames as images
        frames_dir = output_base / f"{output_prefix}_pca_frames"
        os.makedirs(str(frames_dir), exist_ok=True)
        for t in range(T_l):
            frame = pca_spatial[t]  # (W_l, H_l, 3)

            # Convert to uint8 and save as image
            frame_uint8 = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame_uint8)
            frame_path = frames_dir / f"frame_{t:04d}.png"
            img.save(str(frame_path.resolve()))

            # Commit the OutputPath
            frame_path.commit()

    # Create individual component visualizations with mean and std
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot individual components with mean, std, and colorbars
    component_names = ["Red", "Green", "Blue"]
    for i in range(3):
        # Compute mean and std across spatial dimensions (axis=1,2) for temporal analysis
        component_spatial_mean = pca_spatial[:, :, :, i].mean(axis=(1, 2))  # Shape: (T_l,)
        component_spatial_std = pca_spatial[:, :, :, i].std(axis=(1, 2))  # Shape: (T_l,)

        # For visualization, we'll plot the temporal evolution of spatial statistics
        time_steps = np.arange(len(component_spatial_mean))

        # Create a line plot showing temporal evolution
        axes[i].plot(time_steps, component_spatial_mean, "b-", linewidth=2, label="Spatial Mean")
        axes[i].fill_between(
            time_steps,
            component_spatial_mean - component_spatial_std,
            component_spatial_mean + component_spatial_std,
            alpha=0.3,
            color="blue",
            label="±1 Std",
        )

        axes[i].set_title(
            f"PCA Component {i + 1} ({component_names[i]})\n"
            f"Explained Var: {pca.explained_variance_ratio_[i]:.3f}\n"
            f"Avg Mean: {component_spatial_mean.mean():.4f}, Avg Std: {component_spatial_std.mean():.4f}"
        )
        axes[i].set_xlabel("Time Steps")
        axes[i].set_ylabel("Spatial Mean Value")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

        # Add statistics text
        axes[i].text(
            0.02,
            0.98,
            f"Range: [{component_spatial_mean.min():.3f}, {component_spatial_mean.max():.3f}]\n"
            f"Temporal Std: {component_spatial_mean.std():.3f}",
            transform=axes[i].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    components_path = output_base / f"{output_prefix}_pca_components.png"
    # Ensure the directory exists before saving
    os.makedirs(str(components_path.parent.resolve()), exist_ok=True)
    plt.savefig(str(components_path.resolve()), dpi=150, bbox_inches="tight")
    plt.close()

    # Commit the OutputPath
    components_path.commit()

    # Save as video (convert RGB frames to tensor)
    pca_video_tensor = torch.from_numpy(pca_spatial).permute(0, 3, 1, 2)  # (T_l, 3, W_l, H_l)

    # Apply temporal upsampling first
    if t_upsample > 1:
        print(
            f"Temporal upsampling PCA video from {pca_video_tensor.shape[0]} to {pca_video_tensor.shape[0] * t_upsample} frames"
        )
        # Interpolate along the temporal dimension using 1D interpolation per spatial position
        T_original, C, H, W = pca_video_tensor.shape
        T_new = T_original * t_upsample

        # Reshape to (C, H, W, T) for easier temporal interpolation
        pca_temp = pca_video_tensor.permute(1, 2, 3, 0)  # (C, H, W, T)
        pca_temp = pca_temp.reshape(C * H * W, T_original)  # (C*H*W, T)

        # Use 1D interpolation along time dimension
        pca_temp_upsampled = torch.nn.functional.interpolate(
            pca_temp.unsqueeze(1),  # (C*H*W, 1, T)
            size=T_new,
            mode="linear",
            align_corners=False,
        ).squeeze(1)  # (C*H*W, T_new)

        # Reshape back to (C, H, W, T_new) then permute to (T_new, C, H, W)
        pca_video_tensor = pca_temp_upsampled.reshape(C, H, W, T_new).permute(3, 0, 1, 2)

    # Apply spatial upsampling
    if h_upsample > 1 or w_upsample > 1:
        print(
            f"Spatial upsampling PCA video from {pca_video_tensor.shape[2:]} to {pca_video_tensor.shape[2] * h_upsample}x{pca_video_tensor.shape[3] * w_upsample}"
        )
        pca_video_tensor = torch.nn.functional.interpolate(
            pca_video_tensor,
            size=(pca_video_tensor.shape[2] * h_upsample, pca_video_tensor.shape[3] * w_upsample),
            mode="bilinear",
            align_corners=False,
        )

    video_path = output_base / f"{output_prefix}_pca_rgb_buffer_size_1.mp4"
    # Ensure the directory exists before saving video
    os.makedirs(str(video_path.parent.resolve()), exist_ok=True)
    save_video(pca_video_tensor, video_path, fps=30)

    print(f"PCA visualization saved and committed to {output_dir}:")
    if save_individual_frames:
        print(f"  - Individual frames: {output_base / f'{output_prefix}_pca_frames'}/ ({T_l} files)")
    print(f"  - Component analysis: {components_path}")
    print(f"  - RGB video: {video_path} (upsampled {t_upsample}x temporal, {h_upsample}x{w_upsample} spatial)")

    return pca, pca_spatial


def main(
    visualize_pca: bool = False,
    save_decoded_video: bool = True,
    save_control_video: bool = False,
    pca_output_prefix: str = "pca_analysis",
    save_individual_pca_frames: bool = False,
    device: str = "auto",
    dtype=torch.bfloat16,
    motion_file: str = None,
    output_video: str = None,
    control_video: str = None,
    output_dir: str = ".",
    pca_h_upsample: int = 8,
    pca_w_upsample: int = 8,
    pca_t_upsample: int = 4,
):
    # Handle device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    vae_config = WanVAEConfig()

    with torch.device(device), default_dtype(dtype):
        vae_model = WanVAE(config=vae_config)

    vae_model.requires_grad_(False).eval()

    # Load skeletons
    if motion_file is None:
        motion_file = InputPath(
            "s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/HG_007_1/control_latents_debug/rubenh_multicam_5/inference/num_steps_25/dit-checkpoint-0002100224-non_ema/famm/giles_outrage.safetensors"
            # "s3://synthesia-rnd-eun1-dataops-prd-datalake/processed/lvm_video_dataset_ingest/finetuning_video_datasets/dit/stock_personal_avatar_01_2025/GeorginaKJ_snapshot_4/version=0/synthesia-rnd-data-ingest-tracking/landmarks/synthesia_id-77515fba-b933-cab9-660b-d7547c3b8c04.safetensors"
        ).resolve()
    else:
        motion_file = InputPath(motion_file).resolve()
    skeletons = load_safetensor_skeletons(landmarks_path=motion_file, indices=None, return_num_bytes_read=False)

    # Resize skeletons
    resize_size = Size(width=1280, height=720)
    skeletons = resize_skeletons(skeletons, resize_size)

    # Crop skeletons
    crop_region = SkeletonRegion.FullBody.name
    crop_size = Size(width=480, height=640)
    crop_type = CropType.SkeletonCrop.name
    if crop_type == CropType.CentreCrop.name:
        skeletons = center_crop_skeletons(skeletons, crop_size)
    elif crop_type == CropType.SkeletonCrop.name:
        bbox_centres_frames = bbox_centers_from_skeletons(
            skeletons, process_individually=False, normalised=False, region=crop_region
        )
        crop_pad_coordinates_frames = get_crop_or_pad_coordinates(
            bbox_centres=bbox_centres_frames,
            crop_size=crop_size,
            resize_size=resize_size,
        )
        skeletons = box_crop_skeletons(skeletons, crop_pad_coordinates_frames)

    # Render skeletons
    skeleton_renderer = DWPoseCPURenderer()
    control_frames = skeleton_renderer.batch_render_skeleton(skeletons)

    # Save control frames before VAE encoding if requested
    if save_control_video:
        # control_frames from batch_render_skeleton is already in (T, C=3, H, W) format
        if control_video is None:
            control_video_path = OutputPath(
                "s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/HG_007_1/control_latents_debug/rubenh_multicam_5_giles_outrage_input_control_frames.mp4"
            )
        else:
            control_video_path = OutputPath(control_video)
        save_video(control_frames, control_video_path, fps=30)

    # Encode motion
    control_frames = control_frames.to(dtype=dtype, device=device)
    control_frames = control_frames.unsqueeze(0).movedim(2, -1)  # (B=1, T, H, W, C=3)

    with torch.no_grad():
        first_control_frame = control_frames[:, :1]  # (B=1, T=1, H, W, C=3)
        remaining_control_frames = control_frames.unfold(dimension=1, size=5, step=4).movedim(
            -1, -4
        )  # (B=1, S, T=5, H, W, C=3)
        batch_size, seq_len = remaining_control_frames.shape[:2]
        remaining_control_frames = remaining_control_frames.flatten(0, 1)  # (B*S, T=5, H, W, C=3)
        first_control_latent = vae_model.encode_for_inference(first_control_frame)  # (B=1, W_l, H_l, C_l)
        remaining_control_latents = []
        assert batch_size == 1, "Batch size must be 1 for this implementation"
        for i in range(seq_len):
            remaining_control_latents.append(
                vae_model.encode_for_inference(remaining_control_frames[i : i + 1])[:, -1:]
            )  # (B=1, T=1, W_l, H_l, C_l)
        remaining_control_latents = torch.cat(remaining_control_latents, dim=1)  # (B=1, T=S, W_l, H_l, C_l)
        control_latents = torch.cat(
            [first_control_latent, remaining_control_latents], dim=1
        )  # (B=1, T=S+1, W_l, H_l, C_l)

    # Perform PCA analysis and visualization
    if visualize_pca:
        pca_model, pca_visualization = visualize_pca_components(
            control_latents,
            pca_output_prefix,
            save_individual_frames=save_individual_pca_frames,
            output_dir=output_dir,
            h_upsample=pca_h_upsample,
            w_upsample=pca_w_upsample,
            t_upsample=pca_t_upsample,
        )

    if save_decoded_video:
        # Decode motion
        decoded_frames = vae_model.decode_for_inference(control_latents, streaming=False)  # (B=1, T, H, W, C=3)

        control_frames = decoded_frames.squeeze(0).movedim(-1, 1)  # (T, C=3, H, W)

        # Save control videos
        if output_video is None:
            video_path = OutputPath(
                "s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/HG_007_1/control_latents_debug/rubenh_multicam_5_giles_outrage_output_control.mp4"
                # "s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/HG_007_1/control_latents_debug/ada_4_output_control.mp4"
            )
        else:
            video_path = OutputPath(output_video)
        save_video(control_frames, video_path, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode motion from skeleton data using VAE and optionally perform PCA analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core functionality arguments
    parser.add_argument(
        "--visualize-pca",
        action="store_true",
        help="Perform PCA analysis on control latents and generate RGB visualizations",
    )
    parser.add_argument("--save-decoded-video", action="store_true", default=True, help="Save the decoded video output")
    parser.add_argument(
        "--no-decoded-video",
        dest="save_decoded_video",
        action="store_false",
        help="Skip saving the decoded video output",
    )
    parser.add_argument(
        "--save-control-video", action="store_true", default=False, help="Save control frames before VAE encoding"
    )

    # PCA specific arguments
    parser.add_argument(
        "--pca-output-prefix", type=str, default="pca_analysis", help="Output prefix for PCA visualization files"
    )
    parser.add_argument(
        "--save-individual-pca-frames",
        action="store_true",
        help="Save individual PCA frames as PNG images (can generate many files)",
    )
    parser.add_argument("--pca-h-upsample", type=int, default=8, help="Height upsampling factor for PCA video")
    parser.add_argument("--pca-w-upsample", type=int, default=8, help="Width upsampling factor for PCA video")
    parser.add_argument("--pca-t-upsample", type=int, default=4, help="Temporal upsampling factor for PCA video")

    # Model and processing arguments
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="Device to use for processing"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type for model inference",
    )

    # Input/Output arguments
    parser.add_argument("--motion-file", type=str, help="Path to input motion file (overrides default in script)")
    parser.add_argument("--output-video", type=str, help="Path to output video file (overrides default in script)")
    parser.add_argument(
        "--control-video", type=str, help="Path to output control video file (overrides default in script)"
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for PCA visualization files")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    args.dtype = dtype_map[args.dtype]

    main(
        visualize_pca=args.visualize_pca,
        save_decoded_video=args.save_decoded_video,
        save_control_video=args.save_control_video,
        pca_output_prefix=args.pca_output_prefix,
        save_individual_pca_frames=args.save_individual_pca_frames,
        device=args.device,
        dtype=args.dtype,
        motion_file=args.motion_file,
        output_video=args.output_video,
        control_video=args.control_video,
        output_dir=args.output_dir,
        pca_h_upsample=args.pca_h_upsample,
        pca_w_upsample=args.pca_w_upsample,
        pca_t_upsample=args.pca_t_upsample,
    )
