import cv2


def global_threshold(
    gray,
    threshold=127,
    max_value=255
):
    """
    Apply different global thresholding techniques.

    Parameters:
        gray (numpy.ndarray): Grayscale image.
        threshold (int): Threshold value.
        max_value (int): Maximum pixel value.

    Returns:
        dict: Thresholded images.
    """

    results = {}

    _, results["binary"] = cv2.threshold(
        gray,
        threshold,
        max_value,
        cv2.THRESH_BINARY
    )

    _, results["binary_inv"] = cv2.threshold(
        gray,
        threshold,
        max_value,
        cv2.THRESH_BINARY_INV
    )

    _, results["truncate"] = cv2.threshold(
        gray,
        threshold,
        max_value,
        cv2.THRESH_TRUNC
    )

    _, results["tozero"] = cv2.threshold(
        gray,
        threshold,
        max_value,
        cv2.THRESH_TOZERO
    )

    _, results["tozero_inv"] = cv2.threshold(
        gray,
        threshold,
        max_value,
        cv2.THRESH_TOZERO_INV
    )

    print("[INFO] Global thresholding completed.")
    print(f"[INFO] Threshold value: {threshold}")

    return results
