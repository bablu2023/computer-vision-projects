import cv2


def compare_matching_methods(image, template):
    """
    Compare all OpenCV template matching methods.

    Parameters:
        image (numpy.ndarray): Target BGR image.
        template (numpy.ndarray): Grayscale template.

    Returns:
        dict: Matching results for all methods.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    methods = {
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF": cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
    }

    results = {}

    print("\n" + "=" * 70)
    print("Template Matching Method Comparison")
    print("=" * 70)

    for name, method in methods.items():

        response = cv2.matchTemplate(
            gray,
            template,
            method
        )

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
            response
        )

        if method in (
            cv2.TM_SQDIFF,
            cv2.TM_SQDIFF_NORMED
        ):
            score = min_val
            location = min_loc
            best = "Minimum"
        else:
            score = max_val
            location = max_loc
            best = "Maximum"

        print(f"{name:22} Score: {score:10.4f}   Best: {best}")

        results[name] = {
            "response": response,
            "score": score,
            "location": location,
            "method": method
        }

    print("=" * 70)

    return results
