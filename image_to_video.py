from datalib import InputPath, OutputPath
import imageio


def img_to_webm_imageio(input_image_path, output_video_path, duration=5, fps=30):
    """
    Convert a JPG image to WebM video using imageio.

    Args:
        input_image_path (str): Path to input JPG image
        output_video_path (str): Path for output WebM video
        duration (int): Duration of video in seconds (default: 5)
        fps (int): Frames per second (default: 30)
    """
    # Read the image
    img = imageio.imread(input_image_path)

    # Create writer object with WebM codec
    writer = imageio.get_writer(output_video_path, fps=fps, codec='libvpx-vp9')

    # Calculate total frames needed
    total_frames = duration * fps

    # Write the same frame multiple times
    for _ in range(total_frames):
        writer.append_data(img)

    # Close the writer
    writer.close()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    image_dir = InputPath("s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/sidit")

    # Define common image file extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]

    # Collect all image files
    image_files = []
    for extension in image_extensions:
        image_files.extend(list(image_dir.glob(extension)))
        # Also search case-insensitive versions
        image_files.extend(list(image_dir.glob(extension.upper())))

    print(f"Found {len(image_files)} image files in {image_dir}")

    # Process each image file
    for image_file in image_files:
        print(f"Processing: {image_file}")

        # Get the image file name without extension using datalib
        image_name = image_file.stem

        # Create output video path with .webm extension
        output_video_path = OutputPath(f"s3://synthesia-rnd-eun1-experiments/ditwo/kaiwen/sidit/{image_name}.webm")

        # Convert image to WebM
        try:
            # OutputPath.resolve() will handle directory creation automatically
            img_to_webm_imageio(str(image_file.resolve()), str(output_video_path.resolve()))
            # Commit the video to S3
            output_video_path.commit()
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    print("Finished processing all images!")
