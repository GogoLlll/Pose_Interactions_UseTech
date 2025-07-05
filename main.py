import cv2
import json
import numpy as np
from config import input_video, output_video, json_file
from detector import detect_people
from pose_estimator import get_pose_from_roi
from tracker_module import update_tracks
from draw_utils import draw_skeleton
from behavior_analyzer import analyze_behavior

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

track_data = {}
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_people(frame)
    tracks = update_tracks(detections, frame=frame)
    current_tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, width - 1)
        y2 = min(y2, height - 1)
        area = (x2 - x1) * (y2 - y1)
        if area > 50000:
            continue

        roi = frame[y1:y2, x1:x2]
        keypoints = get_pose_from_roi(roi)
        if keypoints:
            draw_skeleton(frame, keypoints, x1, y1)
        else:
            keypoints = [[-1, -1] for _ in range(17)]

        current_tracks.append({
            "id": int(track_id),
            "bbox": [x1, y1, x2, y2],
            "keypoints": keypoints,
            "flags": {}
        })

    # Анализ парного взаимодействия с передачей frame_idx
    interactions = analyze_behavior(current_tracks, width, height, frame_idx)

    # Отображение
    for track in current_tracks:
        track_id = track["id"]
        x1, y1, x2, y2 = track["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'ID {track_id}'

        for interaction in interactions:
            id1, id2 = interaction["pair"]
            if track_id in (id1, id2):
                other_id = id2 if track_id == id1 else id1
                other_track = next(t for t in current_tracks if t["id"] == other_id)
                x1_other, y1_other, x2_other, y2_other = other_track["bbox"]
                center1 = (x1 + x2) // 2, (y1 + y2) // 2
                center2 = (x1_other + x2_other) // 2, (y1_other + y2_other) // 2

                if interaction["type"] == "handshake":
                    label += " (Handshake)"
                    cv2.line(frame, center1, center2, (255, 0, 255), 2)  # Розовая линия
                elif interaction["type"] == "hug":
                    label += " (Hug)"
                    cv2.line(frame, center1, center2, (255, 255, 0), 2)  # Голубая линия

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    timestamp = round(frame_idx / fps, 2)
    track_data[f"frame_{frame_idx:05d}"] = {
        "timestamp": timestamp,
        "people": current_tracks,
        "interactions": interactions
    }

    out.write(frame)
    frame_idx += 1
    cv2.imshow("Tracking + Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

with open(json_file, 'w') as f:
    json.dump(track_data, f, indent=2)

print(f"[DONE] Видео: {output_video}")
print(f"[DONE] JSON: {json_file}")