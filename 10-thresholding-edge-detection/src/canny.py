import cv2


def canny_edge_detection(
    gray,
    low_threshold=100,
    high_threshold=200
):
    """
    Apply Canny edge detection.

    Parameters:
        gray (numpy.ndarray): Grayscale image.
        low_threshold (int): Lower hysteresis threshold.
        high_threshold (int): Upper hysteresis threshold.

    Returns:
        numpy.ndarray: Canny edge image.
    """

    edges = cv2.Canny(
        gray,
        low_threshold,
        high_threshold
    )

    print("[INFO] Canny edge detection completed.")
    print(f"[INFO] Low threshold : {low_threshold}")
    print(f"[INFO] High threshold: {high_threshold}")

    return edges
