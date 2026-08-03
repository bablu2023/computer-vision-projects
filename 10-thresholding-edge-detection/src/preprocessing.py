import cv2


def preprocess_image(
    image,
    blur=True,
    kernel_size=(5, 5)
):
    """
    Convert image to grayscale and optionally apply Gaussian blur.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        blur (bool): Apply Gaussian blur.
        kernel_size (tuple): Gaussian kernel size.

    Returns:
        numpy.ndarray: Preprocessed grayscale image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if blur:
        gray = cv2.GaussianBlur(
            gray,
            kernel_size,
            0
        )
        print("[INFO] Gaussian blur applied.")

    print("[INFO] Grayscale conversion completed.")

    return gray
