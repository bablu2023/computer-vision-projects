def lowe_ratio_test(matches, ratio=0.75):
    """
    Apply Lowe's Ratio Test.

    Parameters:
        matches (list): Output of knnMatch(k=2).
        ratio (float): Ratio threshold.

    Returns:
        list: Filtered good matches.
    """

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    print("[INFO] Lowe's Ratio Test completed.")
    print(f"[INFO] Ratio threshold : {ratio}")
    print(f"[INFO] Raw matches      : {len(matches)}")
    print(f"[INFO] Good matches     : {len(good_matches)}")

    return good_matches
