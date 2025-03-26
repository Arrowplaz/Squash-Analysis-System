import cv2
import os
from utils import (read_video, save_video, get_user_selected_points,
get_user_selected_roi, detect_score, analyze_scoreboard, preprocess_scores, create_heatmap, map_detections,
overlay_heatmap
)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtDetector
import uuid
import numpy as np
import time
import pickle
court_keypoints = []



def process_video(video_path):
    print('Opening Video')
    base_name = os.path.basename(video_path)
    file_name, _ = os.path.splitext(base_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldnt Open Video")
        return

    ret, first_frame = cap.read()
    if not ret:
        print('Error: Could not read first frame')
        return
    
    print('Detecting Court Keypoints')
    court_keypoints = get_user_selected_points(first_frame)

    print('Creating Trackers')
    player_tracker = PlayerTracker('./models/yolov8x.pt')

    player_detections = []
    output_video_frames = []
    frame_idx = 0


    'Stubs'
    player_stub=f'./tracker_stubs/{file_name}/_player.pk1'
    player_exists = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f'Processing frame: {frame_idx}')

        player_detection = player_tracker.detect_frame(frame)
        player_detections.append(player_detection)
        player_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)
        output_frame = player_tracker.draw_bbox(frame, player_detections[-1])
        output_video_frames.append(output_frame)
        frame_idx += 1
    cap.release()

    # if player_stub and not player_exists:
    #     stub_dir = os.path.dirname(player_stub)
    #     if stub_dir and not os.path.exists(stub_dir):
    #         os.makedirs(stub_dir, exist_ok=True)
    #     with open(player_stub, 'wb') as f:
    #         pickle.dump(player_detections, f)

    print('Heatmap Generating....')
    court_keypoints = list(zip(court_keypoints[::2], court_keypoints[1::2]))
    warped_image, overlay, H = create_heatmap(first_frame, court_keypoints)
    mapped_detections = map_detections(player_detections, H)
    heatmap = overlay_heatmap(overlay, mapped_detections)
    
    save_dir = f"./heatmaps/{file_name}"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    heatmap_path = os.path.join(save_dir, f"heatmap_{timestamp}.png")
    cv2.imwrite(heatmap_path, heatmap)
    
    save_path = f"./output_videos/{file_name}"
    os.makedirs(save_path, exist_ok=True)
    final_video_path = os.path.join(save_path, "output.avi")
    print(f"Saving final video to: {final_video_path}")
    save_video(output_video_frames, final_video_path)




    


if __name__ == '__main__':
    process_video("./input_videos/test.mp4")
