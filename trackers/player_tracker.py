from ultralytics import YOLO
import cv2
import pickle
import sys
import os
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance
import numpy as np

class PlayerTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.previous_player_dict = {}
        self.main_ids = []
        self.previous_shirt_colors = {}

    def choose_and_filter_players(self, player_detections, court_keypoints):
        if not player_detections:
            return []

        latest_detection = player_detections[-1]

        if len(self.main_ids) < 2:
            chosen_players = self.choose_players(court_keypoints, latest_detection)
            self.main_ids = list(chosen_players.keys())
            self.previous_shirt_colors = {pid: chosen_players[pid]['shirt_color'] for pid in self.main_ids}
            self.previous_player_dict = {pid: chosen_players[pid] for pid in self.main_ids}
            return [{pid: chosen_players[pid]['bbox'] for pid in self.main_ids}]

        # Upgrade: Match players based on motion + color
        matched_players = self.match_players_by_motion_and_color(
            self.previous_player_dict, latest_detection)

        # Update previous for next frame
        self.previous_player_dict = matched_players
        self.previous_shirt_colors = {pid: matched_players[pid]['shirt_color'] for pid in matched_players}

        player_detections[-1] = {pid: info['bbox'] for pid, info in matched_players.items()}
        return player_detections

    def match_players_by_motion_and_color(self, previous_players, current_detections, max_distance=150, color_weight=0.4):
        if not previous_players:
            return current_detections

        matched_players = {}
        unmatched_detections = set(current_detections.keys())

        for prev_id, prev_info in previous_players.items():
            best_score = float('inf')
            best_detection = None

            prev_center = get_center_of_bbox(prev_info['bbox'])
            prev_color = np.array(prev_info['shirt_color'])

            for det_id in unmatched_detections:
                det_info = current_detections[det_id]
                det_center = get_center_of_bbox(det_info['bbox'])
                det_color = np.array(det_info['shirt_color'])

                motion_dist = measure_distance(prev_center, det_center)
                if motion_dist > max_distance:
                    continue

                color_diff = np.linalg.norm(prev_color - det_color)
                score = (1 - color_weight) * motion_dist + color_weight * color_diff

                if score < best_score:
                    best_score = score
                    best_detection = det_id

            if best_detection is not None:
                matched_players[prev_id] = current_detections[best_detection]
                unmatched_detections.remove(best_detection)

        return matched_players

    def color_distance(self, color1, color2):
        color1 = np.array(color1)
        color2 = np.array(color2)
        return np.linalg.norm(color1 - color2)

    def choose_players(self, court_keypoints, player_dict):
        distances = []

        if not player_dict:
            return {}

        court_center_x = (court_keypoints[0] + court_keypoints[2]) / 2
        court_center_y = (court_keypoints[1] + court_keypoints[5]) / 2
        court_center = (court_center_x, court_center_y)

        for track_id, items in player_dict.items():
            bbox = items['bbox']
            shirt_color = items['shirt_color']

            player_center = get_center_of_bbox(bbox)
            distance = measure_distance(player_center, court_center)

            distances.append((track_id, bbox, shirt_color, distance))

        distances.sort(key=lambda x: x[3])

        chosen_players = {}
        for i in range(min(2, len(distances))):
            chosen_players[distances[i][0]] = {"bbox": distances[i][1], "shirt_color": distances[i][2]}

        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        counter = 0
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            print(f"PLAYER PROGRESS: {counter}/{len(frames)}")
            counter += 1
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir, exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, conf=0.7)[0]
        id_name_dict = results.names
        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                shirt_color = self.extract_shirt_color(frame, result)
                player_dict[track_id] = {"bbox": result, "shirt_color": shirt_color}

        return player_dict

    def extract_shirt_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1

        shirt_region = frame[y1:y1 + height // 2, x1:x2]

        if shirt_region.size == 0:
            return (0, 0, 0)

        shirt_region_small = cv2.resize(shirt_region, (20, 20))

        avg_color = shirt_region_small.mean(axis=0).mean(axis=0)

        return tuple(map(int, avg_color))

    def draw_bbox(self, frame, player_detection):
        for track_id, bbox in player_detection.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if track_id == 1:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        return frame

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if track_id == 1:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
