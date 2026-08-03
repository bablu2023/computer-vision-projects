import cv2


def erode_image(
    image,
    kernel,
    iterations=1
):
    """
    Apply morphological erosion.

    Parameters:
        image (numpy.ndarray): Binary input image.
        kernel (numpy.ndarray): Structuring element.
        iterations (int): Number of erosion iterations.

    Returns:
        numpy.ndarray: Eroded image.
    """

    eroded = cv2.erode(
        image,
        kernel,
        iterations=iterations
    )

    print("[INFO] Erosion applied.")
    print(f"[INFO] Iterations: {iterations}")

    return eroded
