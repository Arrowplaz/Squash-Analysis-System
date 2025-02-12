import cv2
import os
from utils import (read_video, save_video, get_user_selected_points,
get_user_selected_roi, detect_score, analyze_scoreboard, preprocess_scores, create_court_heatmap
)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtDetector
from mini_court import MiniCourt
import uuid
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
    court_detector = CourtDetector()
    court_keypoints = court_detector.detect_court(first_frame, read_from_stub = False, stub_path= f'./tracker_stubs/{file_name}/_court.pk1')
    return 
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
    print(player_detections)

    create_court_heatmap(player_detections, court_keypoints)
    

    



    

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

    # hm = MiniCourt(video_frames[0])
    # player_mini_court_detections= hm.convert_bounding_boxes_to_mini_court_coordinates(player_detections, court_keypoints)
    # #Draw Minicourt
    # output_video_frames = hm.draw_mini_court(output_video_frames)
    # output_video_frames = hm.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    
    #Convert Positions to mini-court positions
    



    final_video_path = os.path.join(save_path, "bad_ball.avi")
    print(f"Saving final video to: {final_video_path}")  # Debugging output
    save_video(output_video_frames, final_video_path)




    


if __name__ == '__main__':
    process_video("./input_videos/Veer Point.mov")


