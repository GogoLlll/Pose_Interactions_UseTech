import torch
import cv2
from ultralytics import YOLO
import numpy as np
from config import keypoint_threshpoint, device

yolo_pose = YOLO("yolo11x-pose.pt").to(device)

def get_pose_from_roi(roi):
    if roi.shape[0] < 30 or roi.shape[1] < 30:
        return None
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        result = yolo_pose(roi_rgb, device=device, imgsz=256, verbose=False)[0]
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        kp = result.keypoints.data[0].cpu().numpy()
        keypoints = []
        for x_kp, y_kp, c_kp in kp:
            if c_kp > keypoint_threshpoint:
                keypoints.append([int(x_kp), int(y_kp)])
            else:
                keypoints.append([-1, -1])
        return keypoints
    return None
