"""
Example script demonstrating how to use the satellite image analysis modules.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_tif_file, load_and_save_tif_as_image
from src.visualization.band_viz import visualize_all_bands, visualize_band_combinations
from src.models.clip_model import CLIPModel
from src.models.dinov2_model import DINOv2Model


def example_load_and_visualize():
    """Example: Load a TIF file and visualize its bands"""
    print("=" * 60)
    print("Example 1: Load and Visualize Satellite Image")
    print("=" * 60)

    # Path to your TIF file
    tif_path = "/path/to/your/image.tif"

    # Load the TIF file
    data, metadata = load_tif_file(tif_path)
    print(f"\nLoaded image:")
    print(f"  Shape: {data.shape}")
    print(f"  Metadata: {metadata}")

    # Visualize all bands
    print("\nVisualizing all bands...")
    visualize_all_bands(tif_path, show_plot=True)

    # Visualize band combinations
    print("\nVisualizing band combinations...")
    visualize_band_combinations(tif_path, show_plot=True)


def example_convert_to_png():
    """Example: Convert TIF to PNG"""
    print("\n" + "=" * 60)
    print("Example 2: Convert TIF to PNG")
    print("=" * 60)

    tif_path = "/path/to/your/image.tif"
    output_path = "./output_truecolor.png"

    # Convert and save
    image_data, metadata = load_and_save_tif_as_image(tif_path, output_path)
    print(f"\nConverted image saved to: {output_path}")


def example_clip_similarity():
    """Example: Use CLIP to compare images"""
    print("\n" + "=" * 60)
    print("Example 3: CLIP Similarity Analysis")
    print("=" * 60)

    # Initialize CLIP model
    clip_model = CLIPModel(device="cuda", gpu_id="0")

    # Compare two images
    image1_path = "/path/to/image1.png"
    image2_path = "/path/to/image2.png"

    similarity = clip_model.compare_images(image1_path, image2_path)
    print(f"\nCosine similarity between images: {similarity:.4f}")

    # Batch similarity analysis
    image_dir = "/path/to/image/directory"
    results = clip_model.batch_similarity(image_dir, K=10, patch_size=64)

    print(f"\nBatch analysis results:")
    print(f"  Overall mean: {results['overall_mean']:.4f}")
    print(f"  Overall std: {results['overall_std']:.4f}")


def example_dinov2_similarity():
    """Example: Use DINOv2 to compare images"""
    print("\n" + "=" * 60)
    print("Example 4: DINOv2 Similarity Analysis")
    print("=" * 60)

    # Initialize DINOv2 model
    dinov2_model = DINOv2Model(model_name="dinov2_vitl14", device="cuda", gpu_id="1")

    # Compare two images
    image1_path = "/path/to/image1.png"
    image2_path = "/path/to/image2.png"

    similarity = dinov2_model.compare_images(image1_path, image2_path)
    print(f"\nDINOv2 Score: {similarity:.4f}")

    # Batch similarity analysis
    image_dir = "/path/to/image/directory"
    results = dinov2_model.batch_similarity(image_dir, K=10, patch_size=64)

    print(f"\nBatch analysis results:")
    print(f"  Overall mean: {results['overall_mean']:.4f}")
    print(f"  Overall std: {results['overall_std']:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Satellite Image Analysis - Example Usage")
    print("=" * 60)

    # Uncomment the examples you want to run
    # Make sure to update the file paths first!

    # example_load_and_visualize()
    # example_convert_to_png()
    # example_clip_similarity()
    # example_dinov2_similarity()

    print("\nNote: Update the file paths in the script before running!")
    print("Uncomment the examples you want to run in the main section.")
