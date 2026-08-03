import cv2


def adaptive_threshold(
    gray,
    max_value=255,
    block_size=11,
    c=2
):
    """
    Apply Adaptive Mean and Adaptive Gaussian Thresholding.

    Parameters:
        gray (numpy.ndarray): Grayscale image.
        max_value (int): Maximum output value.
        block_size (int): Size of local neighborhood (must be odd).
        c (int): Constant subtracted from computed threshold.

    Returns:
        dict: Adaptive threshold results.
    """

    if block_size % 2 == 0:
        raise ValueError(
            "block_size must be an odd number."
        )

    results = {}

    results["adaptive_mean"] = cv2.adaptiveThreshold(
        gray,
        max_value,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )

    results["adaptive_gaussian"] = cv2.adaptiveThreshold(
        gray,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )

    print("[INFO] Adaptive thresholding completed.")
    print(f"[INFO] Block size: {block_size}")
    print(f"[INFO] Constant C: {c}")

    return results
