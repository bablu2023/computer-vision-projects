import cv2


def flip_image(image, flip_code):
    """
    Flip an image.

    Parameters:
        image (numpy.ndarray): Input image.
        flip_code (int):
            0  -> Vertical flip
            1  -> Horizontal flip
           -1  -> Both horizontal and vertical

    Returns:
        numpy.ndarray: Flipped image.
    """

    if flip_code not in [0, 1, -1]:
        raise ValueError(
            "flip_code must be one of: 0 (vertical), 1 (horizontal), -1 (both)."
        )

    flipped = cv2.flip(image, flip_code)

    direction = {
        0: "Vertical",
        1: "Horizontal",
        -1: "Horizontal + Vertical"
    }

    print(f"[INFO] {direction[flip_code]} flip applied.")

    return flipped
