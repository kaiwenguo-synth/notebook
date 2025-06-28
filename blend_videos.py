#!/usr/bin/env python3
"""
Simple Video Alpha Blending Script using ImageIO
Combines two videos using alpha blending: alpha * video1 + (1-alpha) * video2
"""

import imageio
import numpy as np
import argparse
import sys
from PIL import Image
from datalib import InputPath, OutputPath


def resize_frame(frame, target_shape):
    """Resize frame to target shape using PIL."""
    target_height, target_width = target_shape[:2]
    pil_image = Image.fromarray(frame)
    resized = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.array(resized)


def blend_videos(video1_path: InputPath, video2_path: InputPath, output_path: OutputPath, alpha: float = 0.6):
    """Blend two videos using imageio frame-by-frame processing.

    Args:
        video1_path: Path to the first video (overlay)
        video2_path: Path to the second video (background)
        output_path: Path for the output blended video
        alpha: Blending factor (0.0 to 1.0, default 0.6)
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Resolve paths
    video1_local = str(video1_path.resolve())
    video2_local = str(video2_path.resolve())
    output_local = str(output_path.resolve())

    print(f"Input 1: {video1_path}")
    print(f"Input 2: {video2_path}")
    print(f"Output: {output_path}")
    print(f"Alpha: {alpha}")

    try:
        # Open video readers
        reader1 = imageio.get_reader(video1_local)
        reader2 = imageio.get_reader(video2_local)

        # Get video properties
        fps1 = reader1.get_meta_data().get("fps", 30)
        fps2 = reader2.get_meta_data().get("fps", 30)

        # Use the higher fps for output to maintain quality
        output_fps = max(fps1, fps2)

        print(f"Video 1: {fps1} fps")
        print(f"Video 2: {fps2} fps")
        print(f"Output: {output_fps} fps")

        # Create video writer
        writer = imageio.get_writer(output_local, fps=output_fps, codec="libx264", quality=8, macro_block_size=None)

        frame_count = 0

        try:
            while True:
                try:
                    # Read frames from both videos
                    frame1 = reader1.get_next_data()
                    frame2 = reader2.get_next_data()

                except IndexError:
                    # End of one or both videos
                    print(f"Reached end of video(s) at frame {frame_count}")
                    break

                # Ensure both frames have the same dimensions
                if frame1.shape != frame2.shape:
                    # Resize frame1 to match frame2
                    frame1 = resize_frame(frame1, frame2.shape)

                # Convert to float for blending
                frame1_float = frame1.astype(np.float32)
                frame2_float = frame2.astype(np.float32)

                # Alpha blending: alpha * frame1 + (1-alpha) * frame2
                blended = alpha * frame1_float + (1 - alpha) * frame2_float

                # Convert back to uint8 and ensure valid range
                blended_frame = np.clip(blended, 0, 255).astype(np.uint8)

                # Write the blended frames
                writer.append_data(blended_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")

        finally:
            # Clean up resources
            reader1.close()
            reader2.close()
            writer.close()

        print(f"Video blending completed! Processed {frame_count} frames.")

        # Commit to final destination
        output_path.commit()
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error during video processing: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Blend two videos using imageio frame-by-frame processing",
        epilog="""
Examples:
  python blend_videos.py video1.mp4 video2.mp4 output.mp4
  python blend_videos.py video1.mp4 video2.mp4 output.mp4 --alpha 0.7
  python blend_videos.py s3://bucket/video1.mp4 s3://bucket/video2.mp4 s3://bucket/output.mp4
        """,
    )

    parser.add_argument("video1", help="Path to video 1 (overlay)")
    parser.add_argument("video2", help="Path to video 2 (background)")
    parser.add_argument("output", help="Path for output video")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha blending factor (0.0-1.0, default: 0.6)")

    args = parser.parse_args()

    # Create InputPath and OutputPath objects
    video1_path = InputPath(args.video1)
    video2_path = InputPath(args.video2)
    output_path = OutputPath(args.output)

    try:
        blend_videos(video1_path, video2_path, output_path, args.alpha)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
