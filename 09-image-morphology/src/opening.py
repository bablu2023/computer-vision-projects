import cv2


def opening_image(
    image,
    kernel,
    iterations=1
):
    """
    Apply morphological opening.

    Opening = Erosion followed by Dilation.

    Parameters:
        image (numpy.ndarray): Binary input image.
        kernel (numpy.ndarray): Structuring element.
        iterations (int): Number of iterations.

    Returns:
        numpy.ndarray: Opened image.
    """

    opened = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations
    )

    print("[INFO] Opening applied.")
    print(f"[INFO] Iterations: {iterations}")

    return opened
