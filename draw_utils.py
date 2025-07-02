import cv2

pose_connect = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, keypoints, offset_x=0, offset_y=0):
    for x, y in keypoints:
        if x != -1:
            cv2.circle(frame, (x + offset_x, y + offset_y), 3, (0, 0, 255), -1)
    for start, end in pose_connect:
        if keypoints[start] != [-1, -1] and keypoints[end] != [-1, -1]:
            p1 = (keypoints[start][0] + offset_x, keypoints[start][1] + offset_y)
            p2 = (keypoints[end][0] + offset_x, keypoints[end][1] + offset_y)
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
