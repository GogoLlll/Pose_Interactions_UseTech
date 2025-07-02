import cv2
import json
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

        roi = frame[y1:y2, x1:x2]
        keypoints = get_pose_from_roi(roi)
        if keypoints:
            draw_skeleton(frame, keypoints, x1, y1)
            flags = analyze_behavior(keypoints)
        else:
            keypoints = [[-1, -1] for _ in range(17)]
            flags = {}

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_tracks.append({
            "id": int(track_id),
            "bbox": [x1, y1, x2, y2],
            "keypoints": keypoints,
            "flags": flags
        })

    timestamp = round(frame_idx / fps, 2)
    track_data[f"frame_{frame_idx:05d}"] = {
        "timestamp": timestamp,
        "people": current_tracks
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
