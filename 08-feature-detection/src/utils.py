def print_feature_statistics(
    name,
    keypoints,
    descriptors
):
    """
    Print feature detection statistics.
    """

    print("\n" + "=" * 50)
    print(f"{name} Statistics")
    print("=" * 50)

    print(f"Keypoints : {len(keypoints)}")

    if descriptors is None:
        print("Descriptors : None")
    else:
        print(f"Descriptor Shape : {descriptors.shape}")

    print("=" * 50)
