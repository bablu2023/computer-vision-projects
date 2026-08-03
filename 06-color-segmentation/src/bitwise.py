import cv2


def bitwise_and(image, mask):
    """
    Apply bitwise AND operation.

    Parameters:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask.

    Returns:
        numpy.ndarray: Result after AND operation.
    """

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    print("[INFO] Bitwise AND applied.")

    return result


def bitwise_or(image1, image2):
    """
    Apply bitwise OR operation.

    Parameters:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        numpy.ndarray: Combined image.
    """

    result = cv2.bitwise_or(
        image1,
        image2
    )

    print("[INFO] Bitwise OR applied.")

    return result


def bitwise_xor(image1, image2):
    """
    Apply bitwise XOR operation.

    Parameters:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        numpy.ndarray: XOR result.
    """

    result = cv2.bitwise_xor(
        image1,
        image2
    )

    print("[INFO] Bitwise XOR applied.")

    return result


def bitwise_not(image):
    """
    Apply bitwise NOT operation.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Inverted image.
    """

    result = cv2.bitwise_not(image)

    print("[INFO] Bitwise NOT applied.")

    return result
