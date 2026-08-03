import cv2


def preprocess_image(
    image,
    blur_kernel=(3, 3)
):
    """
    Convert image to grayscale and apply Gaussian blur.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        blur_kernel (tuple): Gaussian blur kernel.

    Returns:
        tuple:
            gray - Grayscale image
            blurred - Blurred grayscale image
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

    print("[INFO] Image converted to grayscale.")
    print("[INFO] Gaussian blur applied.")

    return gray, blurred
