#!/usr/bin/env python3
"""
Side-by-Side Video Renderer using FFmpeg
Combines two videos by placing them side by side with optional text overlays
"""

import subprocess
import argparse
import sys
import shlex
import os
from pathlib import Path
from datalib import InputPath, OutputPath


def get_video_info(video_path: str):
    """Get video information using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json

        info = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in info["streams"]:
            if stream["codec_type"] == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        return {
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "fps": eval(video_stream.get("r_frame_rate", "30/1")),
            "duration": float(video_stream.get("duration", 0)),
        }
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get video info for {video_path}: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse video info for {video_path}: {e}")


def find_system_font():
    """Find a suitable system font for text rendering."""
    # Common font locations and names
    font_paths = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/system/fonts/Roboto-Regular.ttf",  # Android
        # macOS
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path

    # Try to find any .ttf font in common directories
    font_dirs = [
        "/usr/share/fonts",
        "/System/Library/Fonts",
        "C:/Windows/Fonts",
    ]

    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if file.lower().endswith((".ttf", ".otf")):
                        return os.path.join(root, file)

    return None


def combine_videos_side_by_side_ffmpeg(
    video1_path: InputPath,
    video2_path: InputPath,
    output_path: OutputPath,
    text1: str = "",
    text2: str = "",
    font_size: int = 24,
    font_color: str = "white",
    font_file: str = "",
):
    """Combine two videos side by side using FFmpeg with text overlays.

    Args:
        video1_path: Path to the first video (left side)
        video2_path: Path to the second video (right side)
        output_path: Path for the output combined video
        text1: Text to display on top left of first video
        text2: Text to display on top left of second video
        font_size: Font size for text overlays
        font_color: Color for text overlays
        font_file: Path to font file (auto-detected if empty)
    """

    # Resolve paths
    video1_local = str(video1_path.resolve())
    video2_local = str(video2_path.resolve())
    output_local = str(output_path.resolve())

    print(f"Left video: {video1_path}")
    print(f"Right video: {video2_path}")
    print(f"Output: {output_path}")
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")

    # Find font file if not provided
    if (text1 or text2) and not font_file:
        font_file = find_system_font()
        if not font_file:
            print("Warning: No font file found. Text overlays may not work.")
        else:
            print(f"Using font: {font_file}")

    try:
        # Get video information
        print("Analyzing input videos...")
        info1 = get_video_info(video1_local)
        info2 = get_video_info(video2_local)

        print(
            f"Video 1: {info1['width']}x{info1['height']} @ {info1['fps']:.2f} fps, duration: {info1['duration']:.2f}s"
        )
        print(
            f"Video 2: {info2['width']}x{info2['height']} @ {info2['fps']:.2f} fps, duration: {info2['duration']:.2f}s"
        )

        # Use the higher fps for output
        output_fps = max(info1["fps"], info2["fps"])
        print(f"Output fps: {output_fps:.2f}")

        # Calculate minimum duration to clip output
        min_duration = min(info1["duration"], info2["duration"])
        if min_duration <= 0:
            print("Warning: Could not determine video durations, output may not be clipped properly")
            min_duration = None
        else:
            print(f"Output duration: {min_duration:.2f}s (clipped to shortest video)")

        # Calculate target height (use the larger height)
        target_height = max(info1["height"], info2["height"])

        # Calculate scaled widths to maintain aspect ratio
        width1 = int(info1["width"] * target_height / info1["height"])
        width2 = int(info2["width"] * target_height / info2["height"])

        output_width = width1 + width2

        print(f"Output dimensions: {output_width}x{target_height}")

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output file

        # Input files
        cmd.extend(["-i", video1_local])
        cmd.extend(["-i", video2_local])

        # Build filter complex
        filters = []

        # Scale both videos to target height
        filters.append(f"[0:v]scale={width1}:{target_height}[v0scaled]")
        filters.append(f"[1:v]scale={width2}:{target_height}[v1scaled]")

        # Add text overlays if specified
        v0_final = "v0scaled"
        v1_final = "v1scaled"

        if text1:
            # Escape text for FFmpeg
            escaped_text1 = text1.replace(":", "\\:").replace("'", "\\'")
            font_param = f":fontfile='{font_file}'" if font_file else ""
            filters.append(
                f"[v0scaled]drawtext=text='{escaped_text1}':fontsize={font_size}:fontcolor={font_color}:x=10:y=10{font_param}[v0text]"
            )
            v0_final = "v0text"

        if text2:
            # Escape text for FFmpeg
            escaped_text2 = text2.replace(":", "\\:").replace("'", "\\'")
            font_param = f":fontfile='{font_file}'" if font_file else ""
            filters.append(
                f"[v1scaled]drawtext=text='{escaped_text2}':fontsize={font_size}:fontcolor={font_color}:x=10:y=10{font_param}[v1text]"
            )
            v1_final = "v1text"

        # Combine videos horizontally
        filters.append(f"[{v0_final}][{v1_final}]hstack=inputs=2[output]")

        # Add filter complex to command
        cmd.extend(["-filter_complex", ";".join(filters)])

        # Map output and set codec options
        cmd.extend(["-map", "[output]"])
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-preset", "medium"])
        cmd.extend(["-crf", "23"])
        cmd.extend(["-r", str(output_fps)])

        # Add audio from first video (if present)
        cmd.extend(["-c:a", "aac"])
        cmd.extend(["-map", "0:a?"])  # ? makes it optional

        # Limit output duration to shortest video
        if min_duration is not None:
            cmd.extend(["-t", str(min_duration)])

        # Output file
        cmd.append(output_local)

        print("Running FFmpeg command...")
        print(f"Command: {' '.join(shlex.quote(arg) for arg in cmd)}")

        # Run FFmpeg
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            print("FFmpeg stderr:", process.stderr)
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")

        print("Side-by-side video creation completed!")

        # Commit to final destination
        output_path.commit()
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error during video processing: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Combine two videos side by side using FFmpeg with text overlays",
        epilog="""
Examples:
  python side_by_side_videos.py video1.mp4 video2.mp4 output.mp4
  python side_by_side_videos.py video1.mp4 video2.mp4 output.mp4 --text1 "Camera 1" --text2 "Camera 2"
  python side_by_side_videos.py video1.mp4 video2.mp4 output.mp4 --text1 "Left View" --text2 "Right View" --font-size 32 --font-color yellow
  python side_by_side_videos.py video1.mp4 video2.mp4 output.mp4 --text1 "Cam1" --text2 "Cam2" --font-file /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
  python side_by_side_videos.py s3://bucket/video1.mp4 s3://bucket/video2.mp4 s3://bucket/output.mp4
        """,
    )

    parser.add_argument("video1", help="Path to video 1 (left side)")
    parser.add_argument("video2", help="Path to video 2 (right side)")
    parser.add_argument("output", help="Path for output video")
    parser.add_argument("--text1", default="", help="Text to display on top left of first video")
    parser.add_argument("--text2", default="", help="Text to display on top left of second video")
    parser.add_argument("--font-size", type=int, default=24, help="Font size for text overlays (default: 24)")
    parser.add_argument("--font-color", default="white", help="Font color for text overlays (default: white)")
    parser.add_argument("--font-file", default="", help="Path to font file (auto-detected if not specified)")

    args = parser.parse_args()

    # Create InputPath and OutputPath objects
    video1_path = InputPath(args.video1)
    video2_path = InputPath(args.video2)
    output_path = OutputPath(args.output)

    try:
        combine_videos_side_by_side_ffmpeg(
            video1_path,
            video2_path,
            output_path,
            args.text1,
            args.text2,
            args.font_size,
            args.font_color,
            args.font_file,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
