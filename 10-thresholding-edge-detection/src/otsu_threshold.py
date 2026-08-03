import cv2


def otsu_threshold(
    gray,
    max_value=255
):
    """
    Apply Otsu's automatic thresholding.

    Parameters:
        gray (numpy.ndarray): Grayscale image.
        max_value (int): Maximum output pixel value.

    Returns:
        tuple:
            threshold_value
            binary_image
    """

    threshold_value, binary = cv2.threshold(
        gray,
        0,
        max_value,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("[INFO] Otsu thresholding completed.")
    print(f"[INFO] Computed threshold: {threshold_value:.2f}")

    return threshold_value, binary
