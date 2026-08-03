import cv2
import os


def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    """
    Load an image from disk.

    Parameters:
        image_path (str): Path to image.
        color_mode (int): OpenCV image loading mode.

    Returns:
        numpy.ndarray: Loaded image.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        image_path,
        color_mode
    )

    if image is None:
        raise ValueError(
            f"Unable to load image: {image_path}"
        )

    height, width = image.shape[:2]

    print("[INFO] Image loaded successfully.")
    print(f"[INFO] Path     : {image_path}")
    print(f"[INFO] Size     : {width} x {height}")
    print(f"[INFO] Channels : {1 if len(image.shape)==2 else image.shape[2]}")

    return image
