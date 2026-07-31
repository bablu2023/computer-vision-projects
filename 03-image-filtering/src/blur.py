import cv2


def average_blur(image, kernel_size=(5, 5)):
    """
    Apply average (mean) blur to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (tuple): Size of the averaging kernel.

    Returns:
        numpy.ndarray: Blurred image.
    """

    blurred = cv2.blur(image, kernel_size)

    print(f"[INFO] Average blur applied with kernel size {kernel_size}")

    return blurred
