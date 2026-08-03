import cv2


def localize_template(
    image,
    template,
    result,
    method=cv2.TM_CCOEFF_NORMED
):
    """
    Locate the best template match and draw a bounding box.

    Parameters:
        image (numpy.ndarray): Original BGR image.
        template (numpy.ndarray): Grayscale template.
        result (numpy.ndarray): Template matching response.
        method (int): Matching method used.

    Returns:
        tuple:
            output_image
            best_location
            best_score
    """

    output = image.copy()

    h, w = template.shape[:2]

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in (
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED
    ):
        top_left = min_loc
        score = min_val
    else:
        top_left = max_loc
        score = max_val

    bottom_right = (
        top_left[0] + w,
        top_left[1] + h
    )

    cv2.rectangle(
        output,
        top_left,
        bottom_right,
        (0, 255, 0),
        2
    )

    print("[INFO] Best match localized.")
    print(f"[INFO] Top-left     : {top_left}")
    print(f"[INFO] Bottom-right : {bottom_right}")
    print(f"[INFO] Score        : {score:.4f}")

    return output, top_left, score
