from ultralytics import YOLO
import cv2
import pickle
import sys
import os
import numpy as np
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance

class PlayerTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.previous_player_dict = {}
        self.main_ids = []
        self.previous_shirt_colors = {}
        self.color_history = {}
        self.history_length = 200

    def choose_and_filter_players(self, player_detections, court_keypoints):
        if not player_detections:
            return []

        def is_overlap(bbox1, bbox2):
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2

            # Check if one bbox is to the left, right, above, or below the other
            if x1_max < x2_min or x2_max < x1_min:
                return False  # No horizontal overlap
            if y1_max < y2_min or y2_max < y1_min:
                return False  # No vertical overlap

            return True

        latest_detection = player_detections[-1]
        chosen_players = self.choose_players(court_keypoints, latest_detection)

        if len(self.main_ids) < 2:
            self.main_ids = list(chosen_players.keys())
            self.previous_shirt_colors = {pid: chosen_players[pid]["shirt_color"] for pid in self.main_ids}
            self.color_history = {pid: [chosen_players[pid]["shirt_color"]] for pid in self.main_ids}
            player_detections[-1] = {pid: chosen_players[pid]["bbox"] for pid in self.main_ids}
            return player_detections

        filtered_player_dict = {}
        remain_ids = self.main_ids.copy()

        pid_mapping = {}

        
        for new_pid in chosen_players:
            new_color = chosen_players[new_pid]["shirt_color"]
            distances = {main_id: np.linalg.norm(np.array(new_color) - np.array(self.previous_shirt_colors[main_id])) for main_id in remain_ids}
            closest_main_id = min(distances, key=distances.get)
            remain_ids.remove(closest_main_id)
            pid_mapping[closest_main_id] = new_pid
            filtered_player_dict[closest_main_id] = [chosen_players[new_pid]['bbox'], chosen_players[new_pid]['shirt_color']]
            

        
        if len(filtered_player_dict) == 2:
            bbox_1 = filtered_player_dict[0]['bbox']
            bbox_2 = filtered_player_dict[1]['bbox']

            if not is_overlap(bbox_1, bbox_2):
                for pid in list(filtered_player_dict.keys()):
                    current_color = filtered_player_dict[pid]['shirt_color']
                    self.color_history.setdefault(pid, []).append(current_color)
                    if len(self.color_history[pid]) > self.history_length:
                        self.color_history[pid].pop(0)
                    avg_color = tuple(np.mean(self.color_history[pid], axis=0).astype(int))
                    self.previous_shirt_colors[pid] = avg_color



        # for pid in self.main_ids:
        #     matched_pid = pid
        #     for new_pid, assigned_main_id in pid_mapping.items():
        #         if assigned_main_id == pid:
        #             matched_pid = new_pid
        #             break
            
        #     if matched_pid in chosen_players:
        #         current_color = chosen_players[matched_pid]["shirt_color"]
        #         self.color_history.setdefault(pid, []).append(current_color)
        #         if len(self.color_history[pid]) > self.history_length:
        #             self.color_history[pid].pop(0)
        #         avg_color = tuple(np.mean(self.color_history[pid], axis=0).astype(int))
        #         self.previous_shirt_colors[pid] = avg_color
        #         filtered_player_dict[pid] = chosen_players[matched_pid]["bbox"]

        player_detections[-1] = {pid: items[0] for pid, items in filtered_player_dict.items()}
        return player_detections


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
