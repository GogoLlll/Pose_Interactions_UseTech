def analyze_behavior(keypoints):
    flags = {
        "sitting": False,
        "gesturing": False
    }

    if keypoints[11] != [-1, -1] and keypoints[13] != [-1, -1] and keypoints[15] != [-1, -1]:
        thigh = abs(keypoints[11][1] - keypoints[13][1])
        shin = abs(keypoints[13][1] - keypoints[15][1])
        if thigh < shin * 0.75:
            flags["sitting"] = True

    if keypoints[7][1] != -1 and keypoints[5][1] != -1 and keypoints[7][1] < keypoints[5][1]:
        flags["gesturing"] = True
    if keypoints[8][1] != -1 and keypoints[6][1] != -1 and keypoints[8][1] < keypoints[6][1]:
        flags["gesturing"] = True

    return flags
