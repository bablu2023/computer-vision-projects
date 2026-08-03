import cv2


def template_match(
    image,
    template,
    method=cv2.TM_CCOEFF_NORMED
):
    """
    Perform template matching.

    Parameters:
        image (numpy.ndarray): Target BGR image.
        template (numpy.ndarray): Template image (grayscale).
        method (int): OpenCV template matching method.

    Returns:
        tuple:
            result (numpy.ndarray)
            image_gray (numpy.ndarray)
    """

    image_gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    result = cv2.matchTemplate(
        image_gray,
        template,
        method
    )

    print("[INFO] Template matching completed.")
    print(f"[INFO] Result shape: {result.shape}")
    print(f"[INFO] Matching method: {method}")

    return result, image_gray
