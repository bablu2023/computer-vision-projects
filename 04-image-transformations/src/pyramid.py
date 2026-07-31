import cv2


def gaussian_pyramid(image, levels=3):
    """
    Build a Gaussian Pyramid.

    Parameters:
        image (numpy.ndarray): Input image.
        levels (int): Number of pyramid levels.

    Returns:
        list: List of images in the Gaussian pyramid.
    """

    pyramid = [image]

    current = image

    for level in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)

    print(f"[INFO] Gaussian Pyramid created ({levels + 1} levels).")

    return pyramid


def laplacian_pyramid(image, levels=3):
    """
    Build a Laplacian Pyramid.

    Parameters:
        image (numpy.ndarray): Input image.
        levels (int): Number of pyramid levels.

    Returns:
        list: List of images in the Laplacian pyramid.
    """

    gaussian = gaussian_pyramid(image, levels)

    laplacian = []

    for i in range(levels):
        expanded = cv2.pyrUp(
            gaussian[i + 1],
            dstsize=(gaussian[i].shape[1], gaussian[i].shape[0])
        )

        lap = cv2.subtract(
            gaussian[i],
            expanded
        )

        laplacian.append(lap)

    laplacian.append(gaussian[-1])

    print(f"[INFO] Laplacian Pyramid created ({levels + 1} levels).")

    return laplacian
