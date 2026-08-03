import cv2


def preprocess_image(
    image,
    blur_kernel=(5, 5)
):
    """
    Convert image to grayscale and binary.

    Parameters:
        image (numpy.ndarray): Input image.
        blur_kernel (tuple): Gaussian kernel size.

    Returns:
        tuple:
            gray
            binary
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        blur_kernel,
        0
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("[INFO] Grayscale conversion completed.")
    print("[INFO] Otsu thresholding applied.")

    return gray, binary
