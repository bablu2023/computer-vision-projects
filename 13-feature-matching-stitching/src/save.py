import cv2
import os


def save_image(image, filename, output_dir="output"):
    """
    Save image to output directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(
        output_dir,
        filename
    )

    success = cv2.imwrite(
        filepath,
        image
    )

    if not success:
        raise IOError(
            f"Failed to save {filepath}"
        )

    print(f"[INFO] Saved: {filepath}")

    return filepath
