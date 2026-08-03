import cv2
import numpy as np


def image_statistics(image, name="Image"):
    """
    Compute basic image statistics.

    Parameters:
        image (numpy.ndarray): Grayscale image.
        name (str): Image name.

    Returns:
        dict: Image statistics.
    """

    stats = {
        "name": name,
        "min": int(np.min(image)),
        "max": int(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image))
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"Minimum Intensity : {stats['min']}")
    print(f"Maximum Intensity : {stats['max']}")
    print(f"Mean Intensity    : {stats['mean']:.2f}")
    print(f"Std Deviation     : {stats['std']:.2f}")

    return stats


def compare_images(*images):
    """
    Compare multiple grayscale images.

    Parameters:
        *images: Tuples in the form (name, image)

    Returns:
        list: List of statistics dictionaries.
    """

    results = []

    print("\n" + "=" * 60)
    print("Image Comparison")
    print("=" * 60)

    for name, image in images:
        results.append(
            image_statistics(image, name)
        )

    return results
