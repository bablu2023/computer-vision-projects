from src.rectangle import draw_rectangle
from src.circle import draw_circle
from src.arrow import draw_arrow
from src.text import draw_text


def annotate_image(image):
    """
    Apply multiple annotations to an image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Annotated image.
    """

    output = image.copy()

    # Bounding box
    output = draw_rectangle(
        output,
        top_left=(120, 100),
        bottom_right=(380, 380),
        color=(0, 255, 0),
        thickness=3
    )

    # Center point
    output = draw_circle(
        output,
        center=(250, 240),
        radius=6,
        color=(0, 0, 255),
        thickness=-1
    )

    # Direction arrow
    output = draw_arrow(
        output,
        start_point=(250, 240),
        end_point=(420, 120),
        color=(255, 0, 0),
        thickness=3,
        tip_length=0.2
    )

    # Object label
    output = draw_text(
        output,
        text="Detected Object",
        position=(120, 90),
        font_scale=0.8,
        color=(0, 255, 0),
        thickness=2
    )

    print("[INFO] Image annotation completed.")

    return output
