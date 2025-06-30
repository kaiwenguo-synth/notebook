from datalib import InputPath, OutputPath
import cv2
import torch
from safetensors.torch import load_file


def main():
    video_path = InputPath(
        "s3://synthesia-rnd-eun1-dataops-prd-datalake/processed/lvm_video_dataset_ingest/finetuning_video_datasets/dit/stock_personal_avatar_01_2025/GeorginaKJ_snapshot_4/version=0/synthesia-rnd-videotranscoder/transcoded/synthesia_id-77515fba-b933-cab9-660b-d7547c3b8c04.webm"
    ).resolve()

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get additional video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Video file: {video_path}")
    print(f"Total frames: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration:.2f} seconds")

    # Release the video capture object
    cap.release()

    # Load the safetensor file
    safetensor_path = InputPath(
        "s3://synthesia-rnd-eun1-dataops-prd-datalake/processed/lvm_video_dataset_ingest/finetuning_video_datasets/dit/stock_personal_avatar_01_2025/GeorginaKJ_snapshot_4/version=0/synthesia-rnd-data-ingest-tracking/landmarks/synthesia_id-77515fba-b933-cab9-660b-d7547c3b8c04.safetensors"
    ).resolve()

    print(f"\nLoading safetensor file: {safetensor_path}")

    try:
        tensors = load_file(str(safetensor_path))
        tensor = next(iter(tensors.values()))  # Get the first tensor
        print(f"Loaded tensor: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"Successfully loaded safetensor file")

    except Exception as e:
        print(f"Error loading safetensor file: {e}")


if __name__ == "__main__":
    main()
