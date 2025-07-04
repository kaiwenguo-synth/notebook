#!/usr/bin/env python3
"""
Single Video Text Overlay Renderer using FFmpeg
Adds a text overlay to a single video.
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


def add_text_to_video_ffmpeg(
    video_path: InputPath,
    output_path: OutputPath,
    text: str = "",
    font_size: int = 24,
    font_color: str = "white",
    font_file: str = "",
):
    """Adds a text overlay to a single video using FFmpeg.

    Args:
        video_path: Path to the input video
        output_path: Path for the output video
        text: Text to display on top left of the video
        font_size: Font size for text overlay
        font_color: Color for text overlay
        font_file: Path to font file (auto-detected if empty)
    """

    # Resolve paths
    video_local = str(video_path.resolve())
    output_local = str(output_path.resolve())

    print(f"Input video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Text: '{text}'")

    # Find font file if not provided
    if text and not font_file:
        font_file = find_system_font()
        if not font_file:
            print("Warning: No font file found. Text overlay may not work.")
        else:
            print(f"Using font: {font_file}")

    try:
        # Get video information
        print("Analyzing input video...")
        info = get_video_info(video_local)

        print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.2f} fps, duration: {info['duration']:.2f}s")

        output_fps = info["fps"]
        duration = info["duration"]

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output file

        # Input file
        cmd.extend(["-i", video_local])

        # Build filter complex
        filters = []
        if text:
            # Escape text for FFmpeg
            escaped_text = text.replace(":", "\\:").replace("'", "\\'")
            font_param = f":fontfile='{font_file}'" if font_file else ""
            filters.append(
                f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:x=10:y=10{font_param}"
            )

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Set codec options
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-preset", "medium"])
        cmd.extend(["-crf", "23"])
        cmd.extend(["-r", str(output_fps)])

        # Add audio from the video (if present)
        cmd.extend(["-c:a", "aac"])
        cmd.extend(["-map", "0:v:0"])
        cmd.extend(["-map", "0:a?"])  # ? makes it optional

        # Limit output duration
        if duration > 0:
            cmd.extend(["-t", str(duration)])

        # Output file
        cmd.append(output_local)

        print("Running FFmpeg command...")
        print(f"Command: {' '.join(shlex.quote(arg) for arg in cmd)}")

        # Run FFmpeg
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            print("FFmpeg stderr:", process.stderr)
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")

        print("Text overlay video creation completed!")

        # Commit to final destination
        output_path.commit()
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error during video processing: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Add a text overlay to a single video using FFmpeg.",
        epilog="""
Examples:
  python single_video_text_overlay.py video.mp4 output.mp4 --text "My Video"
  python single_video_text_overlay.py video.mp4 output.mp4 --text "Copyright 2024" --font-size 18 --font-color yellow
  python single_video_text_overlay.py s3://bucket/video.mp4 s3://bucket/output.mp4 --text "From Cloud"
        """,
    )

    parser.add_argument("video", help="Path to input video")
    parser.add_argument("output", help="Path for output video")
    parser.add_argument("--text", default="", help="Text to display on top left of video")
    parser.add_argument("--font-size", type=int, default=24, help="Font size for text overlay (default: 24)")
    parser.add_argument("--font-color", default="white", help="Font color for text overlay (default: white)")
    parser.add_argument("--font-file", default="", help="Path to font file (auto-detected if not specified)")

    args = parser.parse_args()

    # Create InputPath and OutputPath objects
    video_path = InputPath(args.video)
    output_path = OutputPath(args.output)

    try:
        add_text_to_video_ffmpeg(
            video_path,
            output_path,
            args.text,
            args.font_size,
            args.font_color,
            args.font_file,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
