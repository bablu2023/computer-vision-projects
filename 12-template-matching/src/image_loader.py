import cv2
import os


def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    """
    Load the target image.

    Parameters:
        image_path (str): Path to image.
        color_mode (int): OpenCV loading mode.

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

    h, w = image.shape[:2]

    print("[INFO] Target image loaded.")
    print(f"[INFO] Size : {w} x {h}")

    return image
