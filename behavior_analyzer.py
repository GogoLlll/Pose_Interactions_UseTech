import numpy as np
from collections import defaultdict

# Глобальные словари для отслеживания
interaction_start = defaultdict(bool)
pair_history = defaultdict(list)  # Хранит историю пар для каждого ID
trajectory_history = defaultdict(list)  # Хранит историю центров bbox'ов
HISTORY_LENGTH = 5  # Короткая история для стабильности


def analyze_behavior(tracks, frame_width, frame_height, frame_idx):
    """
    Анализирует пары треков для определения рукопожатия и объятий.
    tracks: список треков в текущем кадре, каждый содержит id, bbox, keypoints
    frame_width, frame_height: размеры кадра для нормализации
    frame_idx: индекс текущего кадра
    Возвращает список подтверждённых взаимодействий [{"pair": [id1, id2], "type": str}]
    """
    pair_flags = {}
    pair_distances = {}

    # Обновляем траектории
    for track in tracks:
        track_id = track["id"]
        x1, y1, x2, y2 = track["bbox"]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        trajectory_history[track_id].append(center)
        if len(trajectory_history[track_id]) > HISTORY_LENGTH:
            trajectory_history[track_id].pop(0)

    # Вычисляем расстояния между парами
    for i, track1 in enumerate(tracks):
        for track2 in tracks[i + 1:]:
            id1, id2 = track1["id"], track2["id"]
            x1, y1, x2, y2 = track1["bbox"]
            x1_other, y1_other, x2_other, y2_other = track2["bbox"]
            center1 = (x1 + x2) / 2, (y1 + y2) / 2
            center2 = (x1_other + x2_other) / 2, (y1_other + y2_other) / 2
            bbox_dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
            pair_distances[(id1, id2)] = bbox_dist

    # Ограничиваем анализ близкими парами
    close_pairs = [(id1, id2) for (id1, id2), dist in pair_distances.items() if dist < 400]

    for id1, id2 in close_pairs:
        track1 = next(t for t in tracks if t["id"] == id1)
        track2 = next(t for t in tracks if t["id"] == id2)
        keypoints1, keypoints2 = track1["keypoints"], track2["keypoints"]
        bbox1, bbox2 = track1["bbox"], track2["bbox"]

        # Проверка валидности ключевых точек
        valid_kp1 = sum(1 for kp in keypoints1 if kp != [-1, -1])
        valid_kp2 = sum(1 for kp in keypoints2 if kp != [-1, -1])
        if valid_kp1 < 5 or valid_kp2 < 5:
            continue

        # Извлечение ключевых точек
        nose1, nose2 = keypoints1[0], keypoints2[0]
        left_shoulder1, right_shoulder1 = keypoints1[5], keypoints1[6]
        left_shoulder2, right_shoulder2 = keypoints2[5], keypoints2[6]
        left_wrist1, right_wrist1 = keypoints1[9], keypoints1[10]
        left_wrist2, right_wrist2 = keypoints2[9], keypoints2[10]

        # Проверка ориентации (опционально для рукопожатия)
        orientation_valid = True
        if left_shoulder1 != [-1, -1] and right_shoulder1 != [-1, -1] and left_shoulder2 != [-1,
                                                                                             -1] and right_shoulder2 != [
            -1, -1]:
            vec1 = np.array(right_shoulder1) - np.array(left_shoulder1)
            vec2 = np.array(right_shoulder2) - np.array(left_shoulder2)
            cos_angle = np.dot(vec1, -vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            if cos_angle > 0.6:  # Угол < ~50 градусов
                orientation_valid = True

        # 1. Проверка рукопожатия (только запястья)
        handshake_detected = False
        if orientation_valid:  # Опциональная ориентация
            wrist_pairs = [
                (left_wrist1, right_wrist2),
                (right_wrist1, left_wrist2),
                (left_wrist1, left_wrist2),
                (right_wrist1, right_wrist2)
            ]
            for wrist1, wrist2 in wrist_pairs:
                if wrist1 != [-1, -1] and wrist2 != [-1, -1]:
                    wrist_dist = np.sqrt((wrist1[0] - wrist2[0]) ** 2 + (wrist1[1] - wrist2[1]) ** 2)
                    if wrist_dist < 50:  # Увеличен порог для надёжности
                        handshake_detected = True
                        interaction_start[(id1, id2)] = True
                        break

        # 2. Проверка объятий (только верхние части тела: плечи или носы)
        hug_detected = False
        if orientation_valid:  # Обязательная ориентация для объятий
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
                    if shoulder_dist < 150 or nose_dist < 150:  # Увеличен порог для надёжности
                        hug_detected = True
                        interaction_start[(id1, id2)] = True
                        break

        pair_flags[(id1, id2)] = {
            "handshake": handshake_detected,
            "hug": hug_detected
        }

    # Фильтрация конфликтов с учётом истории пар
    final_interactions = []
    assigned_ids = set()
    for (id1, id2), flags in sorted(pair_flags.items(), key=lambda x: pair_distances[x[0]], reverse=False):
        pair_key = tuple(sorted([id1, id2]))

        # Проверяем историю пар
        prev_pair1 = pair_history[id1] if pair_history[id1] else []
        prev_pair2 = pair_history[id2] if pair_history[id2] else []
        prev_partner1 = prev_pair1[-1][1] if prev_pair1 and len(prev_pair1[-1]) > 1 else None
        prev_partner2 = prev_pair2[-1][1] if prev_pair2 and len(prev_pair2[-1]) > 1 else None

        # Приоритет текущей паре, если она совпадает с предыдущей
        stable_pair = (prev_partner1 == id2 and prev_partner2 == id1) or (prev_partner1 == id2 or prev_partner2 == id1)

        if id1 not in assigned_ids and id2 not in assigned_ids:
            if flags["handshake"] and interaction_start[pair_key]:
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
            elif flags["hug"] and interaction_start[pair_key]:
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

    # Очищаем флаг и историю, если пара больше не взаимодействует
    current_interacting_pairs = {(id1, id2) for id1, id2 in pair_flags if
                                 pair_flags[(id1, id2)]["handshake"] or pair_flags[(id1, id2)]["hug"]}
    for pair_key in list(interaction_start.keys()):
        if pair_key not in current_interacting_pairs:
            interaction_start[pair_key] = False
    for track_id in tracks:
        if not any(track_id["id"] in pair for pair in current_interacting_pairs):
            pair_history[track_id["id"]].clear()

    return final_interactions