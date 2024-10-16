from utils import (read_video,
                   save_video
                   )

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    print('Working')
    #Read in video
    input_video_path = './input_videos/rally.mp4'
    video_frames = read_video(input_video_path)
    
    #Detect players
    player_tracker = PlayerTracker('./models/yolov8x')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='./tracker_stubs/player_detections.pk1')

    #Detect ball
    ball_tracker = BallTracker('./models/ball_best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub = True,
                                                 stub_path ='./tracker_stubs/ball_detections.pk1')
    
    #Detect court keypoints
    court_model_path = './models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)

    #NOT WORKING
    #court_keypoints = court_line_detector.predict(video_frames[0])


    #Draw Player & Ball Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #NOT WORKING
    #Draw Court Keypoints
    #output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Save the video
    save_video(output_video_frames, 'output_videos2/output_video.avi')



if __name__ == '__main__':
    main()