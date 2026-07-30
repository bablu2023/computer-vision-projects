import cv2


def load_image(image_path):
    """
    Load an image from the given file path.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image if successful.
        None: If the image cannot be loaded.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Unable to load image: {image_path}")
        return None

    print(f"[INFO] Image loaded successfully: {image_path}")
    return image
