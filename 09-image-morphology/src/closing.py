import cv2


def closing_image(
    image,
    kernel,
    iterations=1
):
    """
    Apply morphological closing.

    Closing = Dilation followed by Erosion.

    Parameters:
        image (numpy.ndarray): Binary input image.
        kernel (numpy.ndarray): Structuring element.
        iterations (int): Number of iterations.

    Returns:
        numpy.ndarray: Closed image.
    """

    closed = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )

    print("[INFO] Closing applied.")
    print(f"[INFO] Iterations: {iterations}")

    return closed
