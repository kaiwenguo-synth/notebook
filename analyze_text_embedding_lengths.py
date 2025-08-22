#!/usr/bin/env python3
"""
Script to analyze the distribution of text embedding lengths in the ditwo_pretraining_v2_dense_captioning snapshot.

This script:
1. Samples 1K clips from the specified snapshot
2. Generates text embeddings using the T5 text encoder
3. Creates a histogram showing the distribution of embedding lengths
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from datalib import InputPath

from ditwo.configs.config_dataclasses import DiTTrainingConfig, T5TextEncoderConfig
from ditwo.constants import DATASET_INDEXES
from ditwo.data.metadata import DatasetIndex, VideoClipMetadata
from ditwo.models.text_encoder import T5TextEncoder
from ditwo.utils.logging import logger, set_logging_level


def setup_logging():
    """Setup logging configuration."""
    set_logging_level(logging.INFO)


def create_text_encoder() -> T5TextEncoder:
    """Create and initialize the T5 text encoder."""
    logger.info("Initializing T5 text encoder...")
    config = T5TextEncoderConfig()
    text_encoder = T5TextEncoder(config)
    text_encoder.eval()
    logger.info("T5 text encoder initialized successfully")
    return text_encoder


def sample_clips_from_snapshot(snapshot_path: str, num_samples: int = 1000) -> List[VideoClipMetadata]:
    """
    Sample clips from the specified snapshot.

    Args:
        snapshot_path: Path to the dataset snapshot
        num_samples: Number of clips to sample

    Returns:
        List of VideoClipMetadata objects
    """
    logger.info(f"Loading dataset index from: {snapshot_path}")

    # Create a minimal config for dataset initialization
    config = DiTTrainingConfig()
    config.dataset_config.video_index_file = InputPath(snapshot_path)

    # Load the dataset index directly
    logger.info("Loading dataset index (this may take a moment)...")
    dataset_index = DatasetIndex(InputPath(snapshot_path))
    index_df = dataset_index.split("train").index.collect()

    logger.info(f"Dataset contains {len(index_df)} clips")

    # Sample random clips
    if len(index_df) < num_samples:
        logger.warning(f"Dataset only has {len(index_df)} clips, sampling all of them")
        num_samples = len(index_df)

    # Get random indices
    random_indices = random.sample(range(len(index_df)), num_samples)

    sampled_clips = []
    logger.info(f"Sampling {num_samples} clips...")

    for idx in tqdm(random_indices, desc="Sampling clips", unit="clips"):
        # Get the metadata for this clip
        clip_data = index_df[idx : idx + 1]
        metadata = VideoClipMetadata(
            metadata=clip_data,
            data_source=config.dataset_config.preferred_data_source,
            data_region=None,  # Will be set automatically
        )
        sampled_clips.append(metadata)

    return sampled_clips


def analyze_embedding_lengths(clips: List[VideoClipMetadata], text_encoder: T5TextEncoder) -> List[int]:
    """
    Generate text embeddings for the clips and return their lengths.

    Args:
        clips: List of VideoClipMetadata objects
        text_encoder: Initialized T5TextEncoder

    Returns:
        List of embedding lengths
    """
    embedding_lengths = []

    logger.info("Generating text embeddings...")

    for clip in tqdm(clips, desc="Generating embeddings", unit="clips"):
        try:
            # Get the dense captioning text (same as used in training)
            prompt = clip.dense_captioning

            if not prompt or not prompt.strip():
                # Use empty string for clips without captions
                prompt = ""

            # Generate embeddings with max_sequence_length=512 (the cap mentioned)
            with torch.no_grad():
                text_embeddings, text_lengths = text_encoder(prompt, max_sequence_length=512, zero_out_padding=True)

                # text_lengths is a tensor with shape (1,) containing the actual length
                actual_length = text_lengths.item()
                embedding_lengths.append(actual_length)

        except Exception as e:
            logger.warning(f"Error processing clip {clip.clip_id}: {e}")
            # Add 0 for failed clips
            embedding_lengths.append(0)

    return embedding_lengths


def create_histogram(embedding_lengths: List[int], output_path: str = "text_embedding_lengths_histogram.png"):
    """
    Create and save a histogram of embedding lengths.

    Args:
        embedding_lengths: List of embedding lengths
        output_path: Path to save the histogram
    """
    logger.info("Creating histogram...")

    # Calculate statistics
    lengths_array = np.array(embedding_lengths)
    mean_length = np.mean(lengths_array)
    median_length = np.median(lengths_array)
    max_length = np.max(lengths_array)
    min_length = np.min(lengths_array)

    logger.info("Embedding length statistics:")
    logger.info(f"  Mean: {mean_length:.2f}")
    logger.info(f"  Median: {median_length:.2f}")
    logger.info(f"  Min: {min_length}")
    logger.info(f"  Max: {max_length}")
    logger.info(f"  Samples at max length (512): {np.sum(lengths_array == 512)}")

    # Create histogram
    plt.figure(figsize=(12, 8))

    # Use bins that make sense for the range
    bins = np.arange(0, max_length + 10, 10)  # 10-token bins

    plt.hist(embedding_lengths, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)
    plt.axvline(mean_length, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_length:.1f}")
    plt.axvline(median_length, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_length:.1f}")
    plt.axvline(512, color="green", linestyle="-", linewidth=2, label="Max Length Cap: 512")

    plt.xlabel("Text Embedding Length (tokens)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Text Embedding Lengths\n(ditwo_pretraining_v2_dense_captioning snapshot)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"""Statistics (n={len(embedding_lengths)}):
Mean: {mean_length:.1f}
Median: {median_length:.1f}
Min: {min_length}
Max: {max_length}
At cap (512): {np.sum(lengths_array == 512)} ({100 * np.sum(lengths_array == 512) / len(lengths_array):.1f}%)"""

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Histogram saved to: {output_path}")

    # Also save the raw data
    data_output_path = output_path.replace(".png", "_data.txt")
    with open(data_output_path, "w") as f:
        f.write("# Text embedding lengths from ditwo_pretraining_v2_dense_captioning snapshot\n")
        f.write(f"# Total samples: {len(embedding_lengths)}\n")
        f.write(f"# Mean: {mean_length:.2f}\n")
        f.write(f"# Median: {median_length:.2f}\n")
        f.write(f"# Min: {min_length}\n")
        f.write(f"# Max: {max_length}\n")
        f.write("# Embedding lengths (one per line):\n")
        for length in embedding_lengths:
            f.write(f"{length}\n")

    logger.info(f"Raw data saved to: {data_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze text embedding lengths in ditwo dataset")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of clips to sample (default: 1000)")
    parser.add_argument(
        "--output",
        type=str,
        default="text_embedding_lengths_histogram.png",
        help="Output path for histogram (default: text_embedding_lengths_histogram.png)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")

    args = parser.parse_args()

    # Setup
    setup_logging()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get the snapshot path
    snapshot_name = "ditwo_pretraining_v2_dense_captioning"
    if snapshot_name not in DATASET_INDEXES:
        logger.error(f"Snapshot '{snapshot_name}' not found in DATASET_INDEXES")
        return 1

    snapshot_path = DATASET_INDEXES[snapshot_name]
    logger.info(f"Using snapshot: {snapshot_name}")
    logger.info(f"Snapshot path: {snapshot_path}")

    try:
        # Create text encoder
        text_encoder = create_text_encoder()

        # Sample clips
        clips = sample_clips_from_snapshot(snapshot_path, args.num_samples)
        logger.info(f"Successfully sampled {len(clips)} clips")

        # Analyze embedding lengths
        embedding_lengths = analyze_embedding_lengths(clips, text_encoder)

        # Create histogram
        create_histogram(embedding_lengths, args.output)

        logger.info("Analysis completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
