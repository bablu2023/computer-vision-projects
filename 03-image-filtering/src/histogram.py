import cv2
import matplotlib.pyplot as plt


def plot_histogram(image, title="Histogram"):
    """
    Plot the grayscale histogram of an image.

    Parameters:
        image (numpy.ndarray): Input image.
        title (str): Title of the histogram.
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(8, 4))
    plt.plot(histogram, color="black")
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

    print("[INFO] Histogram plotted successfully.")


def equalize_histogram(image):
    """
    Perform histogram equalization on a grayscale image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    equalized = cv2.equalizeHist(gray)

    print("[INFO] Histogram equalization applied.")

    return equalized
