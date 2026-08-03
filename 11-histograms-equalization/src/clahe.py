import cv2


def apply_clahe(
    gray,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
        gray (numpy.ndarray): Grayscale image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of the local grid.

    Returns:
        numpy.ndarray: CLAHE enhanced image.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(gray)

    print("[INFO] CLAHE applied.")
    print(f"[INFO] Clip Limit : {clip_limit}")
    print(f"[INFO] Tile Grid  : {tile_grid_size}")

    return enhanced
