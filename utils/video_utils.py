import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"Opened Video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        print('Failed to read video')
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def filename_parser(file_name):
    sections = file_name.split('_')
    
    P1_name, P2_name = '', ''
    
    idx = 0
    v_found = False
    
    while idx < len(sections) and sections[idx][0] != '#':
        if sections[idx] == 'V':
            v_found = True
        elif not v_found:
            P1_name += (' ' + sections[idx] if P1_name else sections[idx])  # Avoid leading space
        else:
            P2_name += (' ' + sections[idx] if P2_name else sections[idx])  # Avoid leading space
        
        idx += 1
    
   
    country = sections[idx][1:] if idx < len(sections) and sections[idx][0] == '#' else ''
    game_number = sections[idx + 1][-1] if idx + 1 < len(sections) else ''
    skill_level = sections[idx + 2] if idx + 2 < len(sections) else ''
    
    return {
        "Player 1": P1_name,
        "Player 2": P2_name,
        "Country": country,
        "Game Number": game_number,
        "Skill Level": skill_level
    }



