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
                                                     read_from_stub=True,
                                                     stub_path='./tracker_stubs/player_detections.pk1')

    #Detect ball
    print("Creating Ball Tracker")
    ball_tracker = BallTracker('./models/ball_best.pt')
    print("Detecting Ball")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub = True,
                                                 stub_path ='./tracker_stubs/ball_detections.pk1')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    

    #NOT WORKING
    #Detect court keypoints
    # court_model_path = './models/keypoints_model.pth'
    # court_line_detector = CourtLineDetector(court_model_path)

    #NOT WORKING
    #court_keypoints = court_line_detector.predict(video_frames[0])


    #Draw Player & Ball Bounding Boxes
    print("Drawing detections")
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #NOT WORKING
    #Draw Court Keypoints
    #output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Save the video
    print('Saving Video')
    save_video(output_video_frames, './output_videos/output_video2.avi')

def save_video(frames, output_path):
    print('Starting Save')
    # Get the frame size (assuming all frames are the same size)
    height, width, layers = frames[0].shape
    size = (width, height)

    print('Creating Writer')
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, size)

    print('Writing Frames')
    for frame in frames:
        print(frame)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")




if __name__ == '__main__':
    main()