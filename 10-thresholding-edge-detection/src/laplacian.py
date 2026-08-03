import cv2


def laplacian_edge_detection(
    gray,
    kernel_size=3
):
    """
    Apply Laplacian edge detection.

    Parameters:
        gray (numpy.ndarray): Grayscale image.
        kernel_size (int): Aperture size (1, 3, 5, 7).

    Returns:
        numpy.ndarray: Laplacian edge image.
    """

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
        ksize=kernel_size
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    print("[INFO] Laplacian edge detection completed.")
    print(f"[INFO] Kernel size: {kernel_size}")

    return laplacian
