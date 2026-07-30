"""
Project: Image Viewer
Author: Bablu Kumar Sah

Workflow:
1. Load an image
2. Display image information
3. Convert image to grayscale
4. Save the grayscale image
"""

from src.image_loader import load_image
from src.image_info import print_image_info
from src.image_converter import convert_to_grayscale
from src.image_saver import save_image


def main():
    image_path = "images/sample.jpg"
    output_path = "output/sample_gray.jpg"

    # Step 1: Load image
    image = load_image(image_path)

    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    # Step 2: Print image information
    print_image_info(image)

    # Step 3: Convert to grayscale
    gray_image = convert_to_grayscale(image)

    # Step 4: Save grayscale image
    save_image(output_path, gray_image)

    print("\nImage processing completed successfully.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
