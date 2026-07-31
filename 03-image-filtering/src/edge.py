import cv2


def canny_edge(image, threshold1=100, threshold2=200):
    """
    Apply Canny edge detection.

    Parameters:
        image (numpy.ndarray): Input image.
        threshold1 (int): Lower threshold for hysteresis.
        threshold2 (int): Upper threshold for hysteresis.

    Returns:
        numpy.ndarray: Edge-detected image.
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(
        gray,
        threshold1,
        threshold2
    )

    print(
        f"[INFO] Canny edge detection applied "
        f"(threshold1={threshold1}, threshold2={threshold2})"
    )

    return edges
