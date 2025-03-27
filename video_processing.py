import cv2
import os
import pickle
import time
import gc
import copy
from utils import (
    get_user_selected_points, create_heatmap, map_detections,
    overlay_heatmap, save_video
)
from trackers import PlayerTracker
import numpy as np

def process_video(video_path):
    print('Opening Video')
    base_name = os.path.basename(video_path)
    file_name, _ = os.path.splitext(base_name)

    # Check if the video has already been processed by looking for the detection files or heatmap
    detections_path = f"./detections/{file_name}"
    print(detections_path)
    print(video_path)
    heatmap_save_dir = f"./heatmaps/{file_name}"

    if os.path.exists(detections_path):
        print(f"Heatmap for {file_name} already exists. Skipping video processing.")
        # Skip the video processing and go directly to heatmap generation
        all_detections = []
        for file in sorted(os.listdir(detections_path)):
            if file.endswith('.pkl'):
                with open(os.path.join(detections_path, file), 'rb') as f:
                    all_detections.extend(pickle.load(f))

        print('Generating Heatmap...')
        first_frame = cv2.imread(f'./input_videos/{file_name}.jpg')  # Assuming the first frame was saved as a .jpg or use other methods
        court_keypoints = get_user_selected_points(first_frame)  # If needed, this can be pre-saved as well
        court_keypoints = list(zip(court_keypoints[::2], court_keypoints[1::2]))
        warped_image, overlay, H = create_heatmap(first_frame, court_keypoints)
        mapped_detections = map_detections(all_detections, H)
        heatmap = overlay_heatmap(overlay, mapped_detections)

        save_dir = f"./heatmaps/{file_name}"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        heatmap_path = os.path.join(save_dir, f"heatmap_{timestamp}.png")
        cv2.imwrite(heatmap_path, heatmap)
        print(f"Heatmap saved to: {heatmap_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print('Error: Could not read first frame.')
        return

    print('Detecting Court Keypoints')
    court_keypoints = get_user_selected_points(first_frame)

    print('Creating Trackers')
    player_tracker = PlayerTracker('./models/yolov8x.pt')

    # Set up video writer
    save_path = f"./output_videos/{file_name}"
    os.makedirs(save_path, exist_ok=True)
    final_video_path = os.path.join(save_path, "output.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(final_video_path, fourcc, fps, (width, height))

    os.makedirs(detections_path, exist_ok=True)

    player_detections = []
    frame_idx = 0
    chunk_size = 1000  # Save every 1000 frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f'Processing frame: {frame_idx}')

        # Process and track players
        detections = player_tracker.detect_frame(frame)
        player_detections.append(detections)
        filtered_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)

        output_frame = player_tracker.draw_bbox(frame, filtered_detections[-1])
        out.write(output_frame)  # Write frame directly to video

       

    # Periodically save detections to disk and free memory
    if frame_idx % chunk_size == 0 and frame_idx > 0:
        chunk_file = os.path.join(detections_path, f"detections_{frame_idx}.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(player_detections, f)
        
        tmp = copy.deepcopy(player_detections[-1])  # Ensure a full copy
        player_detections.clear()  # Free memory
        gc.collect()
        player_detections.append(tmp)  # Restore the last frame


        frame_idx += 1

    cap.release()
    out.release()
    print(f"Final video saved to: {final_video_path}")

    # Load all detections for heatmap
    print('Loading saved detections...')
    all_detections = []
    for file in sorted(os.listdir(detections_path)):
        if file.endswith('.pkl'):
            with open(os.path.join(detections_path, file), 'rb') as f:
                all_detections.extend(pickle.load(f))

    print('Generating Heatmap...')
    court_keypoints = list(zip(court_keypoints[::2], court_keypoints[1::2]))
    warped_image, overlay, H = create_heatmap(first_frame, court_keypoints)
    mapped_detections = map_detections(all_detections, H)
    heatmap = overlay_heatmap(overlay, mapped_detections)

    save_dir = f"./heatmaps/{file_name}"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    heatmap_path = os.path.join(save_dir, f"heatmap_{timestamp}.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Heatmap saved to: {heatmap_path}")

if __name__ == '__main__':
    process_video("./input_videos/test.mp4")
