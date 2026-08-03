import cv2


def create_kernel(
    shape="rect",
    size=(5, 5)
):
    """
    Create a morphological structuring element.

    Parameters:
        shape (str): rect, ellipse, or cross.
        size (tuple): Kernel size.

    Returns:
        numpy.ndarray: Structuring element.
    """

    shape = shape.lower()

    if shape == "rect":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            size
        )

    elif shape == "ellipse":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            size
        )

    elif shape == "cross":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            size
        )

    else:
        raise ValueError(
            "Shape must be 'rect', 'ellipse', or 'cross'."
        )

    print(f"[INFO] Kernel created: {shape}")
    print(f"[INFO] Kernel size: {size}")

    return kernel
