from utils import (read_video,
                   save_video
                   )

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    print('Opening Video')
    #Read in video
    input_video_path = './input_videos/rally.mp4'
    print('Video Opened! Begining to read')
    video_frames = read_video(input_video_path)
    
    
    #Detect players
    print("Creating Player Tracker")
    player_tracker = PlayerTracker('./models/yolov8x.pt')
    print("Detecting Players")
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub = True,
                                                     stub_path='./tracker_stubs/player_detections.pk1')

    #Detect ball
    print("Creating Ball Tracker")
    ball_tracker = BallTracker('./models/ball_best.pt')
    print("Detecting Ball")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub = True,
                                                 stub_path ='./tracker_stubs/ball_detections_2.pk1')
    print('Filling in gaps')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    

    #Detect court keypoints
    print('Finding court')
    court_model_path = './models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    # court_keypoints = court_line_detector.predict(video_frames[0])
    court_keypoints = [556, 629, 1405, 627, 334, 1073, 1630, 1072]

    #Filter out non-player detections
    print('Filtering Players')
    player_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)


    #Draw Player & Ball Bounding Boxes
    print("Drawing detections")
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #NOT WORKING
    #Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    ##Draw frame number in video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    #Save the video
    print('Saving Video')
    save_video(output_video_frames, "output_videos/output_video.avi")







if __name__ == '__main__':
    main()