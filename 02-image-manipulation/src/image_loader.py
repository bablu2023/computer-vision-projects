import cv2


def load_image(image_path):
    """
    Load an image from disk.

    Parameters:
        image_path (str): Path to the image.

    Returns:
        numpy.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    print(f"[INFO] Image loaded successfully: {image_path}")

    return image
