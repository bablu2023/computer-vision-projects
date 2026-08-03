import cv2


def find_contours(binary_image):
    """
    Find contours in a binary image.

    Parameters:
        binary_image (numpy.ndarray): Binary threshold image.

    Returns:
        tuple:
            contours (list): Detected contours.
            hierarchy (numpy.ndarray): Contour hierarchy.
    """

    contours, hierarchy = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"[INFO] Contours detected: {len(contours)}")

    return contours, hierarchy
