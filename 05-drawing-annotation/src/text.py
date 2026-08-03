import cv2


def draw_text(
    image,
    text="OpenCV",
    position=(50, 50),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(255, 255, 255),
    thickness=2,
    line_type=cv2.LINE_AA
):
    """
    Draw text on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        text (str): Text to display.
        position (tuple): Bottom-left corner of the text (x, y).
        font (int): OpenCV font type.
        font_scale (float): Font size.
        color (tuple): Text color in BGR format.
        thickness (int): Text thickness.
        line_type (int): Line type.

    Returns:
        numpy.ndarray: Image with text drawn.
    """

    output = image.copy()

    cv2.putText(
        output,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        line_type
    )

    print(f"[INFO] Text added: '{text}'")

    return output
