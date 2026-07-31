import cv2
import matplotlib.pyplot as plt


def display_image(title, image):
    """
    Display an image using Matplotlib.

    Parameters:
        title (str): Window title.
        image (numpy.ndarray): Input image.
    """

    # Convert BGR (OpenCV) to RGB (Matplotlib)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.show()
