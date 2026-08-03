import cv2
import numpy as np


def color_histogram(image):
    """
    Compute BGR color histograms.

    Parameters:
        image (numpy.ndarray): Input BGR image.

    Returns:
        dict: Histograms for Blue, Green, and Red channels.
    """

    channels = ("blue", "green", "red")
    histograms = {}

    for i, channel in enumerate(channels):

        histogram = cv2.calcHist(
            [image],
            [i],
            None,
            [256],
            [0, 256]
        )

        histograms[channel] = histogram

        print(f"[INFO] {channel.capitalize()} histogram computed.")
        print(f"[INFO] Shape: {histogram.shape}")
        print(f"[INFO] Total pixels: {int(np.sum(histogram))}")

    return histograms
