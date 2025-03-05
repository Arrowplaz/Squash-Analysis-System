import cv2
import os
from utils import (read_video, save_video, get_user_selected_points,
get_user_selected_roi, detect_score, analyze_scoreboard, preprocess_scores, create_heatmap, map_detections,
overlay_heatmap
)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtDetector
from mini_court import MiniCourt
import uuid
import numpy as np
import time
court_keypoints = []



def process_video(video_path):

    
    print('Opening Video')
    # Read in video
    base_name = os.path.basename(video_path)
    file_name, _ = os.path.splitext(base_name)
    video_frames = read_video(video_path)

    first_frame = video_frames[0]
    # Get user-selected ROI for the scoreboard
    print("Getting scoreboard ROI")
    # get_user_selected_roi(first_frame)

    #Detect the court
    print("Detecting Court Keypoints")
    # court_detector = CourtDetector()
    # court_keypoints = court_detector.detect_court(first_frame, read_from_stub = False, stub_path= f'./tracker_stubs/{file_name}/_court.pk1', show=True)
    court_keypoints = get_user_selected_points(first_frame)
    print(court_keypoints)
    # Initialize player tracker
    print("Creating Player Tracker")
    player_tracker = PlayerTracker('./models/yolov8x.pt')
    print("Detecting Players")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path=f'./tracker_stubs/{file_name}/_player.pk1')

    # Initialize ball tracker
    print("Creating Ball Tracker")
    ball_tracker = BallTracker('./models/ball_best.pt')
    print("Detecting Ball")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path=f'./tracker_stubs/{file_name}/_ball.pk1')
    print('Filling in gaps')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Filter out non-player detections based on court keypoints
    print('Filtering Players')
    player_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)


    court_keypoints = list(zip(court_keypoints[::2], court_keypoints[1::2]))
    warped_image, overlay, H = create_heatmap(first_frame, court_keypoints)
    mapped_detections = map_detections(player_detections, H)
    heatmap = overlay_heatmap(overlay, mapped_detections)
    # Ensure directory exists
    save_dir = f"./heatmaps/{file_name}"
    os.makedirs(save_dir, exist_ok=True)

    


    # Use timestamp to avoid overwriting files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    heatmap_path = os.path.join(save_dir, f"heatmap_{timestamp}.png")
    cv2.imwrite(heatmap_path, heatmap)

    # # Prepare a padded warped image so that it aligns with the overlay's dimensions:
    # padding = 0
    # # The overlay canvas is larger by the padding amount
    # padded_warped = np.zeros_like(overlay)
    # # Resize the warped image to exactly match the court area (without padding)
    # overlay_height = overlay.shape[0] - padding
    # overlay_width = overlay.shape[1] - padding
    # resized_warped = cv2.resize(warped_image, (overlay_width, overlay_height))
    # # Place the resized warped image into the padded canvas at the correct offset
    # padded_warped[padding:, padding:] = resized_warped

    # # Blend the padded warped image and the overlay into a composite image
    # composite = cv2.addWeighted(padded_warped, 0.7, overlay, 0.3, 0)

    # for (x, y) in mapped_detections:
    #     x, y = int(x), int(y)  # Ensure coordinates are integers
    #     cv2.circle(composite, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green circles for players


    # Display the composite image in a single window
    # cv2.imshow("Warped Image with Overlay", composite)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # Analyze scoreboard
    # print("Analyzing Scoreboard")
    # scores = analyze_scoreboard(video_frames)
    # scores = preprocess_scores(scores)
    # print(f"Detected Scores: {scores}")

    save_path = f"./output_videos/{file_name}"

    # Create all intermediate directories if they don't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # # Segment video by points based on scores
    # print("Segmenting Video by Points")
    # score_frames = list(scores.keys())
    # for idx in range(len(score_frames)):
    #     start_frame, end_frame = score_frames[idx], score_frames[idx + 1]
    #     point_video_frames = video_frames[start_frame:end_frame]
    #     # Draw Player & Ball Bounding Boxes
    #     print("Drawing detections")
    #     output_video_frames = player_tracker.draw_bboxes(point_video_frames, player_detections)
    #     print(f'Saving Video: {idx + 1}')
    #     save_video(point_video_frames, f"{save_path}/point_{idx + 1}.avi")
    
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

   
    



    final_video_path = os.path.join(save_path, "output.avi")
    print(f"Saving final video to: {final_video_path}")  # Debugging output
    save_video(output_video_frames, final_video_path)




    


if __name__ == '__main__':
    process_video("./input_videos/Veer Point.mov")


