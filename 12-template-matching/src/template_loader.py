import cv2
import os


def load_template(template_path):
    """
    Load the template image.

    Parameters:
        template_path (str): Template image path.

    Returns:
        numpy.ndarray: Grayscale template.
    """

    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Template not found: {template_path}"
        )

    template = cv2.imread(
        template_path,
        cv2.IMREAD_GRAYSCALE
    )

    if template is None:
        raise ValueError(
            f"Unable to load template: {template_path}"
        )

    h, w = template.shape

    print("[INFO] Template loaded.")
    print(f"[INFO] Size : {w} x {h}")

    return template
