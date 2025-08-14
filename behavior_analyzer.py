import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

interaction_start = defaultdict(bool)
pair_history = defaultdict(list) 
trajectory_history = defaultdict(list) 
interaction_frame_count = defaultdict(int)  
HISTORY_LENGTH = 5 
MIN_INTERACTION_FRAMES = 3 


def is_stationary(track_id, threshold):
    if len(trajectory_history[track_id]) < HISTORY_LENGTH:
        return False
    centers = np.array(trajectory_history[track_id])
    max_dist = np.max(np.sqrt(np.sum(np.diff(centers, axis=0) ** 2, axis=1)))
    return max_dist < threshold


def analyze_behavior(tracks, frame_width, frame_height, frame_idx):
    pair_flags = {}
    pair_distances = {}

    for track in tracks:
        track_id = track["id"]
        x1, y1, x2, y2 = track["bbox"]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        trajectory_history[track_id].append(center)
        if len(trajectory_history[track_id]) > HISTORY_LENGTH:
            trajectory_history[track_id].pop(0)

    for i, track1 in enumerate(tracks):
        for track2 in tracks[i + 1:]:
            id1, id2 = track1["id"], track2["id"]
            nose1, nose2 = track1["keypoints"][0], track2["keypoints"][0]
            eye_left1, eye_right1 = track1["keypoints"][1], track1["keypoints"][2]
            eye_left2, eye_right2 = track2["keypoints"][1], track2["keypoints"][2]

            if nose1 != [-1, -1] and nose2 != [-1, -1]:
                nose_dist = np.sqrt((nose1[0] - nose2[0]) ** 2 + (nose1[1] - nose2[1]) ** 2)
            else:
                x1, y1, x2, y2 = track1["bbox"]
                x1_other, y1_other, x2_other, y2_other = track2["bbox"]
                head1 = (x1 + x2) / 2, y1
                head2 = (x1_other + x2_other) / 2, y1_other
                nose_dist = np.sqrt((head1[0] - head2[0]) ** 2 + (head1[1] - head2[1]) ** 2)

            orientation_valid = False
            if nose1 != [-1, -1] and nose2 != [-1, -1] and \
               eye_left1 != [-1, -1] and eye_right1 != [-1, -1] and \
               eye_left2 != [-1, -1] and eye_right2 != [-1, -1]:
                mid_eye1 = np.mean([eye_left1, eye_right1], axis=0)
                mid_eye2 = np.mean([eye_left2, eye_right2], axis=0)
                face_vec1 = np.array(nose1) - mid_eye1
                face_vec2 = np.array(nose2) - mid_eye2
                cos_angle = np.dot(face_vec1, -face_vec2) / (np.linalg.norm(face_vec1) * np.linalg.norm(face_vec2) + 1e-6)
                orientation_valid = cos_angle > 0.8 
            else:
                orientation_valid = nose_dist < 0.2 * frame_width 

            logger.info(f"Pair {id1, id2}: nose_dist={nose_dist:.2f}, orientation_valid={orientation_valid}")

            if nose_dist < 0.2 * frame_width and orientation_valid:
                pair_distances[(id1, id2)] = nose_dist
            else:
                pair_distances[(id1, id2)] = float('inf')

    close_pairs = [(id1, id2) for (id1, id2), dist in pair_distances.items() if dist < 0.2 * frame_width]

    for id1, id2 in close_pairs:
        track1 = next(t for t in tracks if t["id"] == id1)
        track2 = next(t for t in tracks if t["id"] == id2)
        keypoints1, keypoints2 = track1["keypoints"], track2["keypoints"]
        bbox1, bbox2 = track1["bbox"], track2["bbox"]

        valid_kp1 = sum(1 for kp in keypoints1 if kp != [-1, -1])
        valid_kp2 = sum(1 for kp in keypoints2 if kp != [-1, -1])
        if valid_kp1 < 5 or valid_kp2 < 5:
            continue

        nose1, nose2 = keypoints1[0], keypoints2[0]
        left_shoulder1, right_shoulder1 = keypoints1[5], keypoints1[6]
        left_shoulder2, right_shoulder2 = keypoints2[5], keypoints2[6]
        left_wrist1, right_wrist1 = keypoints1[9], keypoints1[10]
        left_wrist2, right_wrist2 = keypoints2[9], keypoints2[10]

        orientation_valid = True
        if left_shoulder1 != [-1, -1] and right_shoulder1 != [-1, -1] and left_shoulder2 != [-1, -1] and right_shoulder2 != [-1, -1]:
            vec1 = np.array(right_shoulder1) - np.array(left_shoulder1)
            vec2 = np.array(right_shoulder2) - np.array(left_shoulder2)
            cos_angle = np.dot(vec1, -vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            orientation_valid = cos_angle > 0.6 

        handshake_detected = False
        if orientation_valid:
            wrist_pairs = [
                (left_wrist1, right_wrist2),
                (right_wrist1, left_wrist2),
                (left_wrist1, left_wrist2),
                (right_wrist1, right_wrist2)
            ]
            for wrist1, wrist2 in wrist_pairs:
                if wrist1 != [-1, -1] and wrist2 != [-1, -1]:
                    wrist_dist = np.sqrt((wrist1[0] - wrist2[0]) ** 2 + (wrist1[1] - wrist2[1]) ** 2)
                    if wrist_dist < 0.03 * frame_width: 
                        handshake_detected = True
                        interaction_start[(id1, id2)] = True
                        interaction_frame_count[(id1, id2)] += 1
                        logger.info(f"Handshake detected for pair {id1, id2}: wrist_dist={wrist_dist:.2f}")
                        break

        hug_detected = False
        if orientation_valid:
            shoulder_pairs = [
                (left_shoulder1, left_shoulder2),
                (left_shoulder1, right_shoulder2),
                (right_shoulder1, left_shoulder2),
                (right_shoulder1, right_shoulder2)
            ]
            nose_dist = float('inf')
            if nose1 != [-1, -1] and nose2 != [-1, -1]:
                nose_dist = np.sqrt((nose1[0] - nose2[0]) ** 2 + (nose1[1] - nose2[1]) ** 2)
            for shoulder1, shoulder2 in shoulder_pairs:
                if shoulder1 != [-1, -1] and shoulder2 != [-1, -1]:
                    shoulder_dist = np.sqrt((shoulder1[0] - shoulder2[0]) ** 2 + (shoulder1[1] - shoulder2[1]) ** 2)
                    if shoulder_dist < 0.1 * frame_width or nose_dist < 0.1 * frame_width: 
                        hug_detected = True
                        interaction_start[(id1, id2)] = True
                        interaction_frame_count[(id1, id2)] += 1
                        logger.info(f"Hug detected for pair {id1, id2}: shoulder_dist={shoulder_dist:.2f}, nose_dist={nose_dist:.2f}")
                        break

        pair_flags[(id1, id2)] = {
            "handshake": handshake_detected,
            "hug": hug_detected
        }

    final_interactions = []
    assigned_ids = set()
    for (id1, id2), flags in sorted(pair_flags.items(), key=lambda x: pair_distances[x[0]], reverse=False):
        pair_key = tuple(sorted([id1, id2]))

        prev_pair1 = pair_history[id1] if pair_history[id1] else []
        prev_pair2 = pair_history[id2] if pair_history[id2] else []
        prev_partner1 = prev_pair1[-1][1] if prev_pair1 and len(prev_pair1[-1]) > 1 else None
        prev_partner2 = prev_pair2[-1][1] if prev_pair2 and len(prev_pair2[-1]) > 1 else None

        stable_pair = (prev_partner1 == id2 and prev_partner2 == id1) or (prev_partner1 == id2 or prev_partner2 == id1)

        if id1 not in assigned_ids and id2 not in assigned_ids:
            if flags["handshake"] and interaction_start[pair_key] and interaction_frame_count[pair_key] >= MIN_INTERACTION_FRAMES:
                if stable_pair or not any(
                        prev_partner1 == other_id or prev_partner2 == other_id for other_id in assigned_ids):
                    final_interactions.append({"pair": [id1, id2], "type": "handshake"})
                    assigned_ids.add(id1)
                    assigned_ids.add(id2)
                    pair_history[id1].append((frame_idx, id2))
                    pair_history[id2].append((frame_idx, id1))
                    if len(pair_history[id1]) > HISTORY_LENGTH:
                        pair_history[id1].pop(0)
                    if len(pair_history[id2]) > HISTORY_LENGTH:
                        pair_history[id2].pop(0)
            elif flags["hug"] and interaction_start[pair_key] and interaction_frame_count[pair_key] >= MIN_INTERACTION_FRAMES:
                if stable_pair or not any(
                        prev_partner1 == other_id or prev_partner2 == other_id for other_id in assigned_ids):
                    final_interactions.append({"pair": [id1, id2], "type": "hug"})
                    assigned_ids.add(id1)
                    assigned_ids.add(id2)
                    pair_history[id1].append((frame_idx, id2))
                    pair_history[id2].append((frame_idx, id1))
                    if len(pair_history[id1]) > HISTORY_LENGTH:
                        pair_history[id1].pop(0)
                    if len(pair_history[id2]) > HISTORY_LENGTH:
                        pair_history[id2].pop(0)

    current_interacting_pairs = {(id1, id2) for id1, id2 in pair_flags if any(
        pair_flags[(id1, id2)][flag] for flag in ["handshake", "hug"])}
    for pair_key in list(interaction_start.keys()):
        if pair_key not in current_interacting_pairs:
            interaction_start.pop(pair_key, None)
            interaction_frame_count.pop(pair_key, None)
    for track_id in tracks:
        if not any(track["id"] in pair for pair in current_interacting_pairs):
            pair_history.pop(track["id"], None)

    return final_interactions
