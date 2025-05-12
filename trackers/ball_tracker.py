from ultralytics import YOLO
from inference import get_model
import cv2
import pickle
import pandas as pd
import numpy as np
import supervision as sv
from utils import ultralytics_key


class BallTracker:
    def __init__(self, model_path):
        self.model = get_model(model_id="ball-detection-vamqx/8", api_key=ultralytics_key)

    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions



    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            
            return ball_detections

        counter = 1
        for frame in frames:
            print(f"BALL PROGRESS: {counter}/{len(frames)}")
            counter += 1
            ball_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections
    
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.25)[0]
        detections = sv.Detections.from_inference(results)

        ball_dict = {}
        for box in detections.xyxy:
            result = box.tolist()
            ball_dict[1] = result
        
        return ball_dict

        


    def draw_bbox(self, video_frame, ball_detection):
   
        for track_id, bbox in ball_detection.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        return video_frame
