import cv2
import os


def save_image(image, filename, output_dir="output"):
    """
    Save an image to the specified output directory.

    Parameters:
        image (numpy.ndarray): Image to save.
        filename (str): Name of the output file.
        output_dir (str): Directory where the image will be saved.

    Returns:
        str: Full path of the saved image.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    success = cv2.imwrite(output_path, image)

    if success:
        print(f"[INFO] Image saved: {output_path}")
    else:
        raise IOError(f"Failed to save image: {output_path}")

    return output_path
