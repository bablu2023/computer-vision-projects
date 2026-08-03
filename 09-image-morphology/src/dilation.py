import cv2


def dilate_image(
    image,
    kernel,
    iterations=1
):
    """
    Apply morphological dilation.

    Parameters:
        image (numpy.ndarray): Binary input image.
        kernel (numpy.ndarray): Structuring element.
        iterations (int): Number of dilation iterations.

    Returns:
        numpy.ndarray: Dilated image.
    """

    dilated = cv2.dilate(
        image,
        kernel,
        iterations=iterations
    )

    print("[INFO] Dilation applied.")
    print(f"[INFO] Iterations: {iterations}")

    return dilated
