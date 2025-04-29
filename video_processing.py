import cv2
import os
import pickle
import time
import gc
import copy
from utils import (
    get_user_selected_points, create_heatmap, map_detections,
    overlay_heatmap, save_video, parse_file_name, insert_match,
    filename_parser, get_user_selected_roi, detect_score, compute_distance_traveled
)
from trackers import PlayerTracker
import numpy as np

def process_video(video_path):
    os.environ['TESSDATA_PREFIX'] = '/home/anagireddygari/tessdata'
    print('Opening Video')
    base_name = os.path.basename(video_path)
    file_name, _ = os.path.splitext(base_name)

    # Check if the video has already been processed by looking for the detection files or heatmap
    detections_path = f"./detections/{file_name}"
    heatmap_save_dir = f"./heatmaps/{file_name}"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print('Error: Could not read first frame.')
        return

    print('Detecting Court Keypoints')
    # court_keypoints = get_user_selected_points(first_frame)
    court_keypoints = [722, 462, 1297, 462, 385, 777, 1650, 797]
    print("Court Keypoints: ", court_keypoints)

    score_points = (912, 961, 97, 36)
    # print('Select Scoreboard')
    scoreboard_keypoints = get_user_selected_roi(first_frame)
    print('Scoreboard ROI: ', scoreboard_keypoints)
    return
    print('Creating Trackers')
    player_tracker = PlayerTracker('./models/yolov8x.pt')

    # Set up video writer
    save_path = f"./output_videos/{file_name}"
    os.makedirs(save_path, exist_ok=True)
    final_video_path = os.path.join(save_path, f"{file_name}.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter(final_video_path, fourcc, fps, (width, height))

    os.makedirs(detections_path, exist_ok=True)

    force_check = False

    player_detections = []
    score_detections = []  # List to store score detections
    prev_p1_score = 0
    prev_p2_score = 0
    last_winner = None

    frame_idx = 0
    chunk_size = 1000  # Save every 1000 frames
    if os.listdir(detections_path) == []:
        while cap.isOpened():
            # if frame_idx == 1000:
            #     break
            ret, frame = cap.read()
            if not ret:
                break

            print(f'Processing frame: {frame_idx}')

            # Process and track players
            detections = player_tracker.detect_frame(frame)
            player_detections.append(detections)
            filtered_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)
            player_detections = filtered_detections
            # output_frame = player_tracker.draw_bbox(frame, player_detections[-1])
            # out.write(output_frame)  # Write frame directly to video

            #Detect scoreboard
            if frame_idx % 1000 == 0 or force_check:
                if force_check:
                    print("FORCE CHECKING")
                try:
                    player1_score, player2_score = detect_score(frame)                    
                    if player1_score is not None and player2_score is not None:
                        force_check = False
                        player1_score, player2_score = int(player1_score), int(player2_score)
                        valid_increment = (
                            (player1_score == prev_p1_score + 1 and player2_score == prev_p2_score) or
                            (player2_score == prev_p2_score + 1 and player1_score == prev_p1_score) or
                            (player1_score == prev_p1_score and player2_score == prev_p2_score)
                        )
                        if not valid_increment:
                            raise Exception(f"Not valid increment, previous scores {prev_p1_score}, {prev_p2_score}")

                        print(f"Scores: Player 1 - {player1_score}, Player 2 - {player2_score}")
                        if (player1_score != prev_p1_score) or (player2_score != prev_p2_score):
                            point_winner = None
                            if player1_score != prev_p1_score and player1_score > prev_p1_score:
                                point_winner = 'Player 1'
                            elif player2_score != prev_p2_score and player2_score > prev_p2_score:
                                point_winner = 'Player 2'

                            # Append score detection only when there is a change
                            score_detections.append({
                                'frame_idx': frame_idx,
                                'player1_score': player1_score,
                                'player2_score': player2_score,
                                'point_winner': point_winner
                            })
                            print(f"Point Winner: {point_winner}")

                            # Update previous scores
                            prev_p1_score = player1_score
                            prev_p2_score = player2_score
                            last_winner = point_winner
                            
                    else:
                        force_check = True
                except Exception as e:
                    print(e)
                    print(f"Faulty OCR detection at frame {frame_idx}. Retrying...")
                    force_check = True
                    
            # Periodically save detections to disk and free memory
            if frame_idx % chunk_size == 0 and frame_idx > 0:
                print('Saving')
                chunk_file = os.path.join(detections_path, f"detections_{frame_idx}.pkl")
                with open(chunk_file, 'wb') as f:
                    pickle.dump(player_detections, f)
                
                tmp = copy.deepcopy(player_detections[-1])  # Ensure a full copy
                player_detections.clear()  # Free memory
                gc.collect()
                player_detections.append(tmp)  # Restore the last frame


            frame_idx += 1
        #Make sure last little bit is saved too
        chunk_file = os.path.join(detections_path, f"detections_{frame_idx}.pkl")
        scores_file = os.path.join(detections_path, f"scores.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(player_detections, f)
        with open(scores_file, 'wb') as f:
            pickle.dump(score_detections, f)
        score_detections.clear()
        player_detections.clear()

    


    cap.release()
    # out.release()
    print(f"Final video saved to: {final_video_path}")

    # Load all detections for heatmap
    print('Loading saved detections...')
    all_detections = []
    score_detections = []
    for file in sorted(os.listdir(detections_path)):
        if file.endswith('.pkl') and file != "scores.pkl":
            with open(os.path.join(detections_path, file), 'rb') as f:
                all_detections.extend(pickle.load(f))
        elif file == "scores.pkl":
            with open(os.path.join(detections_path, file), 'rb') as f:
                score_detections.extend(pickle.load(f))
    
    print(score_detections)
    

    print('Generating Heatmap...')
    court_keypoints = list(zip(court_keypoints[::2], court_keypoints[1::2]))
    warped_image, overlay, H = create_heatmap(first_frame, court_keypoints)

    track_ids = []

    for d in all_detections:
        if len(list(d.keys())) == 2:
            track_ids = list(d.keys())
            break

    p1_detections = []
    p2_detections = []
    
    for d in all_detections:
        if track_ids[0] in d:
            p1_detections.append(d[track_ids[0]])
        if track_ids[1] in d:
            p2_detections.append(d[track_ids[1]])

    p1_tmp = [{1: value} for value in p1_detections]
    p2_tmp = [{2: value} for value in p2_detections]
    p1_mapped_detections = map_detections(p1_tmp, H)
    p2_mapped_detections = map_detections(p2_tmp, H)
    
    mapped_detections = map_detections(all_detections, H)
    heatmap = overlay_heatmap(overlay, mapped_detections)

    #Show distance graph
    #compute_distance_traveled(p1_mapped_detections)

    winner = None
    


    print("Uploaded to MongoDB")
    video_data = filename_parser(file_name)
    if prev_p1_score > prev_p2_score:
        winner = video_data['Player 1']
    else:
        winner = video_data['Player 2']
    
    insert_match(video_data['Player 1'], video_data['Player 2'], video_data['Country'], video_data['Game Number'],
                 video_data['Skill Level'], p1_detections, p2_detections, court_keypoints, p1_mapped_detections.tolist(), p2_mapped_detections.tolist(), score_detections, video_data['Gender'], winner)

    print("Uploaded to Mongo")

    save_dir = f"./heatmaps/{file_name}"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    heatmap_path = os.path.join(save_dir, f"heatmap_{timestamp}.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Heatmap saved to: {heatmap_path}")

   


def process_videos_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        print('filename: ', filename)
        if filename.endswith(".mp4") and filename.__contains__('Pro'):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {video_path}")
            process_video(video_path)
            

if __name__ == '__main__':
    input_folder = "./input_videos"
    process_video("./input_videos/Marwan_Elshorbagy_V_Gregoire_Marche_#US_Game1_Pro_M.mp4")
    # process_videos_in_folder(input_folder)
    



   

    
