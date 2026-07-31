import cv2


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        diameter (int): Diameter of the pixel neighborhood.
        sigma_color (float): Filter sigma in the color space.
        sigma_space (float): Filter sigma in the coordinate space.

    Returns:
        numpy.ndarray: Bilateral filtered image.
    """

    filtered = cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )

    print(
        f"[INFO] Bilateral filter applied "
        f"(diameter={diameter}, "
        f"sigma_color={sigma_color}, "
        f"sigma_space={sigma_space})"
    )

    return filtered
