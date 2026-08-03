import cv2


def preprocess_image(
    image,
    blur_kernel=(5, 5)
):
    """
    Convert image to grayscale, apply Gaussian blur,
    Otsu thresholding, and morphological opening.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        blur_kernel (tuple): Gaussian blur kernel size.

    Returns:
        tuple:
            gray      - Grayscale image
            blurred   - Blurred image
            threshold - Binary image
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        blur_kernel,
        0
    )

    # Otsu Thresholding
    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological Opening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    print("[INFO] Image preprocessing completed.")
    print("[INFO] Otsu Thresholding applied.")
    print("[INFO] Morphological opening applied.")

    return gray, blurred, threshold
