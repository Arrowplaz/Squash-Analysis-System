from ultralytics import YOLO
import cv2
import pickle
import sys
import os
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance

class PlayerTracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.previous_player_dict = {}
        self.main_ids = []

    
    def choose_and_filter_players(self, player_detections, court_keypoints):
        if not player_detections:  
            return []

        if len(self.main_ids) < 2:
            chosen_players = self.choose_players(court_keypoints, player_detections[-1])
            self.main_ids = list(chosen_players.keys())
            return [chosen_players]
        
        main_ids = self.main_ids
        chosen_players = self.choose_players(court_keypoints, player_detections[-1])

        filtered_player_dict = {}
        remaining_ids = main_ids.copy()

        for track_id, bbox in chosen_players.items():
            if track_id in main_ids:
                filtered_player_dict[track_id] = bbox
                remaining_ids.remove(track_id)

        # Assign new detections to remaining IDs if needed
        for track_id, bbox in chosen_players.items():
            if not remaining_ids:  
                break
            if track_id not in main_ids:
                filtered_player_dict[remaining_ids.pop(0)] = bbox
        
        player_detections[-1] = filtered_player_dict 
        return player_detections

       

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        
        # If there are no detections, return an empty list
        if not player_dict:
            return {}

        # Calculate the center of the court
        court_center_x = (court_keypoints[0] + court_keypoints[2]) / 2
        court_center_y = (court_keypoints[1] + court_keypoints[5]) / 2
        court_center = (court_center_x, court_center_y)

        # Go through all detections in the current frame
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            # Calculate distance from the player's center to the court center
            distance = measure_distance(player_center, court_center)

            distances.append((track_id, bbox, distance))

        # Sort by minimum distance to the court keypoints
        distances.sort(key=lambda x: x[2])

        chosen_players = {}

        #Grab the track_id and b
        for i in range(min(2, len(distances))):
            chosen_players[distances[i][0]] = distances[i][1]


        return chosen_players




    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
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
    
    def detect_frame(self,frame):
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
                player_dict[track_id] = result
        
        return player_dict

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
            # Draw the bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Corrected putText formatting
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # Corrected rectangle argument formatting
                if track_id == 1:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames

       