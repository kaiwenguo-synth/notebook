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

from datalib import InputPath, OutputPath

from ditwo.data.dataset import bbox_centers_from_skeletons, get_crop_or_pad_coordinates
from ditwo.models.vae.wan import WanVAE
from ditwo.configs.config_dataclasses import SkeletonRegion, WanVAEConfig
from ditwo.utils.ffmpeg import save_video
from ditwo.utils.misc import default_dtype, Size
from ditwo.utils.skeleton.cpu_renderer import DWPoseCPURenderer
from ditwo.utils.skeleton.io import load_safetensor_skeletons
from ditwo.utils.skeleton.transforms import box_crop_skeletons, resize_skeletons


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    vae_config = WanVAEConfig()

    with torch.device(device), default_dtype(dtype):
        vae_model = WanVAE(config=vae_config)

    vae_model.requires_grad_(False).eval()

    # Load skeletons
    motion_file = InputPath(
        # "s3://synthesia-rnd-eun1-experiments/ditwo/HG_007_1/ditwo_pretraining_v2/inference/num_steps_25/rubenh_multicam_5/dit-checkpoint-0002100224-non_ema/famm/giles_outrage.safetensors"
        "s3://synthesia-rnd-eun1-dataops-prd-datalake/processed/lvm_video_dataset_ingest/finetuning_video_datasets/dit/stock_personal_avatar_01_2025/GeorginaKJ_snapshot_4/version=0/synthesia-rnd-data-ingest-tracking/landmarks/synthesia_id-77515fba-b933-cab9-660b-d7547c3b8c04.safetensors"
    ).resolve()
    skeletons = load_safetensor_skeletons(landmarks_path=motion_file, indices=None, return_num_bytes_read=False)

    # Resize skeletons
    resize_size = Size(width=1280, height=720)
    skeletons = resize_skeletons(skeletons, resize_size)

    # Crop skeletons
    crop_region = SkeletonRegion.FullBody.name
    crop_size = Size(width=1280, height=720)
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

    # Encode motion
    control_frames = control_frames.to(dtype=dtype, device=device)
    control_frames = control_frames.unsqueeze(0).movedim(2, -1)  # (B=1, T, H, W, C=3)

    with torch.no_grad():
        control_latents = vae_model.encode_for_inference(control_frames)  # (B=1, T_l, W_l, H_l, C_l)

    # Decode motion
    decoded_frames = vae_model.decode_for_inference(control_latents, streaming=False)  # (B=1, T, H, W, C=3)

    control_frames = decoded_frames.squeeze(0).movedim(-1, 1)  # (T, C=3, H, W)

    # Highlight non-zero pixels
    for frame in control_frames:
        mask = frame < 0.2
        frame[mask] = frame[mask].abs() * 10

    # Save control videos
    video_path = OutputPath(
        # "s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/HG_007_1/control_latents_debug/rubenh_multicam_5_giles_outrage_output_control.mp4"
        "s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/HG_007_1/control_latents_debug/ada_4_output_control.mp4"
    )
    save_video(control_frames, video_path, fps=30)


if __name__ == "__main__":
    main()
