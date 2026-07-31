import cv2


def median_blur(image, kernel_size=5):
    """
    Apply median blur to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the median filter kernel.
                           Must be an odd integer greater than 1.

    Returns:
        numpy.ndarray: Median blurred image.

    Raises:
        ValueError: If kernel_size is not an odd integer > 1.
    """

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd integer greater than 1."
        )

    blurred = cv2.medianBlur(image, kernel_size)

    print(f"[INFO] Median blur applied (kernel={kernel_size})")

    return blurred
