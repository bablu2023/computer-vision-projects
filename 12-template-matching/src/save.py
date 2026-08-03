import os
import cv2


def save_image(
    image,
    filename,
    output_dir="output"
):
    """
    Save an image to disk.

    Parameters:
        image (numpy.ndarray): Image to save.
        filename (str): Output filename.
        output_dir (str): Output directory.

    Returns:
        str: Saved file path.
    """

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    success = cv2.imwrite(filepath, image)

    if not success:
        raise IOError(f"Failed to save image: {filepath}")

    print(f"[INFO] Saved: {filepath}")

    return filepath
