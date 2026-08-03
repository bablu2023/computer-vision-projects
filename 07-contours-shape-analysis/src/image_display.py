import cv2
import matplotlib.pyplot as plt


def display_image(
    image,
    title="Image",
    cmap=None,
    figsize=(6, 6),
    show_axis=False
):
    """
    Display an image using Matplotlib.
    """

    plt.figure(figsize=figsize)

    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap or "gray")
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)

    plt.title(title)

    if not show_axis:
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"[INFO] Displayed: {title}")
