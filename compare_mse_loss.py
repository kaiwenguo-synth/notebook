#!/usr/bin/env python3
"""
Compare MSE loss from two TensorBoard experiments and create a combined visualization.
Supports both local paths and AWS S3 paths via datalib.
"""

from __future__ import annotations

import argparse
import glob
import os

from datalib import InputPath, OutputPath
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def read_tensorboard_logs(log_dir, tag_name="losses/train_mse"):
    """
    Read scalar values from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard event files (as Path object)
        tag_name: Name of the scalar tag to extract (e.g., 'mse_loss', 'train/mse_loss')

    Returns:
        steps: List of step numbers
        values: List of corresponding values
        selected_tag: The actual tag name used
    """
    # Find event files in the directory
    event_files = glob.glob(os.path.join(str(log_dir), "events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    # Use the most recent event file
    event_file = max(event_files, key=os.path.getctime)
    print(f"Reading from: {event_file}")

    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Get available tags
    available_tags = event_acc.Tags()
    print(f"Available scalar tags: {available_tags.get('scalars', [])}")

    # Try to find the MSE loss tag
    mse_tags = [tag for tag in available_tags.get("scalars", []) if "mse" in tag.lower() or "loss" in tag.lower()]

    if not mse_tags:
        raise ValueError(f"No MSE loss tags found. Available tags: {available_tags.get('scalars', [])}")

    # Use the specified tag or the first matching tag
    if tag_name in available_tags.get("scalars", []):
        selected_tag = tag_name
    else:
        selected_tag = mse_tags[0]
        print(f"Tag '{tag_name}' not found. Using '{selected_tag}' instead.")

    # Extract scalar events
    scalar_events = event_acc.Scalars(selected_tag)

    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]

    return steps, values, selected_tag


def create_comparison_tensorboard(
    input1_dir, input2_dir, output_dir, input1_name="Input1", input2_name="Input2", tag_name="losses/train_mse"
):
    """
    Create a new TensorBoard log comparing two experiments.

    Args:
        input1_dir: Path to first tensorboard directory (Path object)
        input2_dir: Path to second tensorboard directory (Path object)
        output_dir: Path to output tensorboard directory (Path object)
        input1_name: Name for first input (for labels)
        input2_name: Name for second input (for labels)
        tag_name: Name of the scalar tag to compare
    """
    # Read data from both experiments
    print(f"\n=== Reading {input1_name} ===")
    steps_1, values_1, actual_tag_1 = read_tensorboard_logs(input1_dir, tag_name)

    print(f"\n=== Reading {input2_name} ===")
    steps_2, values_2, actual_tag_2 = read_tensorboard_logs(input2_dir, tag_name)

    # Create two runs so TensorBoard overlays them in one figure
    run1_dir = os.path.join(str(output_dir), str(input1_name))
    run2_dir = os.path.join(str(output_dir), str(input2_name))
    writer1 = SummaryWriter(run1_dir)
    writer2 = SummaryWriter(run2_dir)

    print(f"\n=== Writing comparison to {output_dir} ===")
    print(f"Run 1: {run1_dir}  tag: {tag_name}")
    print(f"Run 2: {run2_dir}  tag: {tag_name}")

    # Write both curves under the SAME tag but in different runs
    for step, value in zip(steps_1, values_1):
        writer1.add_scalar(tag_name, value, step)

    for step, value in zip(steps_2, values_2):
        writer2.add_scalar(tag_name, value, step)

    # Close the writers
    writer1.close()
    writer2.close()

    print("\nComparison TensorBoard created successfully!")
    print(f"{input1_name}: {len(steps_1)} data points")
    print(f"{input2_name}: {len(steps_2)} data points")
    print("\nTo view the comparison, run:")
    print(f"tensorboard --logdir={output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MSE loss from two TensorBoard experiments. Supports local and S3 paths."
    )
    parser.add_argument(
        "--input1",
        required=True,
        help="Path to first TensorBoard log directory (local or s3://bucket/path/)",
    )
    parser.add_argument(
        "--input2",
        required=True,
        help="Path to second TensorBoard log directory (local or s3://bucket/path/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output TensorBoard directory (local or s3://bucket/path/)",
    )
    parser.add_argument(
        "--name1",
        default="Input1",
        help="Name for first input in the comparison (default: Input1)",
    )
    parser.add_argument(
        "--name2",
        default="Input2",
        help="Name for second input in the comparison (default: Input2)",
    )
    parser.add_argument(
        "--tag",
        default="losses/train_mse",
        help="TensorBoard scalar tag to compare (default: losses/train_mse)",
    )

    args = parser.parse_args()

    try:
        # Use InputPath for inputs - automatically handles S3 downloads
        print("=== Processing input paths ===")
        input1_path = InputPath(args.input1)
        input2_path = InputPath(args.input2)

        # Use OutputPath for output - automatically handles S3 uploads
        output_path = OutputPath(args.output)

        # Resolve paths to local directories
        input1_local = input1_path.resolve()
        input2_local = input2_path.resolve()
        output_local = output_path.resolve()

        # Create the comparison
        create_comparison_tensorboard(input1_local, input2_local, output_local, args.name1, args.name2, args.tag)

        # Commit output to S3 if it's an S3 path
        if args.output.startswith("s3://"):
            print("\n=== Uploading results to S3 ===")
            output_path.commit(compress_folder=False)
            print(f"\nResults uploaded to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
