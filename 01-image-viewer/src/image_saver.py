import cv2
import os


def save_image(output_path, image):
    """
    Save an image to the specified file path.

    Parameters:
        output_path (str): Path where the image will be saved.
        image (numpy.ndarray): Image to save.
    """
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    success = cv2.imwrite(output_path, image)

    if success:
        print(f"[INFO] Image saved successfully: {output_path}")
    else:
        print(f"[ERROR] Failed to save image: {output_path}")
