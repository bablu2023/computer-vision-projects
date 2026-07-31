import cv2


def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (tuple): Size of the Gaussian kernel.
                              Both values must be odd.
        sigma (float): Standard deviation of the Gaussian kernel.
                       If 0, OpenCV calculates it automatically.

    Returns:
        numpy.ndarray: Gaussian blurred image.
    """

    blurred = cv2.GaussianBlur(
        image,
        kernel_size,
        sigma
    )

    print(
        f"[INFO] Gaussian blur applied "
        f"(kernel={kernel_size}, sigma={sigma})"
    )

    return blurred
