import torch
from ultralytics import YOLO
from config import conf_threshold, device

yolo_det = YOLO("weights/yolo11x.pt").to(device)

def detect_people(frame):
    with torch.no_grad():
        results = yolo_det(frame, device=device, imgsz=1280, iou=0.4)[0]  # Используем iou вместо iou_thres
    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf > conf_threshold:
            w, h = x2 - x1, y2 - y1
            area = w * h
            if 1000 < area < 50000:  # Фильтрация по площади bbox
                detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(conf), "person"))
    return detections