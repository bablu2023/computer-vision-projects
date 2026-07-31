import os
import cv2


def save_image(image, filename, output_dir="output"):
    """
    Save an image to disk.

    Parameters:
        image (numpy.ndarray): Image to save.
        filename (str): Name of the output file.
        output_dir (str): Directory where the image will be saved.

    Returns:
        str: Full path of the saved image.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    success = cv2.imwrite(filepath, image)

    if not success:
        raise IOError(f"Failed to save image: {filepath}")

    print(f"[INFO] Image saved: {filepath}")

    return filepath
