import cv2
import re
import easyocr

score_box_coords = []

# Enable GPU for EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

def get_user_selected_roi(frame, meta=None):
    global score_box_coords

    if meta is not None:
        score_box_coords = meta
        return score_box_coords

    print("Select the region of interest (ROI) for the scoreboard.")
    cv2.namedWindow("Select Scoreboard ROI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Select Scoreboard ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    score_box_coords = cv2.selectROI("Select Scoreboard ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return score_box_coords

def detect_score(frame):
    global score_box_coords
    x, y, w, h = score_box_coords
    score_roi = frame[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to detect text
    result = reader.readtext(gray, detail=1)  # We need details for bounding boxes
    print("OCR Result:", result)

    # Initialize variables to store player scores
    player1_score = None
    player2_score = None

    # Manually correct common OCR misinterpretations
    corrected_text = ""
    for text, box in result:
        # Correct the misinterpretations for each piece of text
        text = text.upper()
        text = text.replace('S', '5').replace('O', '0').replace('I', '1')\
                   .replace('L', '1').replace('Z', '2').replace('G', '6')

        corrected_text += text + " "

    print("CORRECTED TEXT:", corrected_text.strip())

    # Keep only digits and dashes
    cleaned_text = re.sub(r'[^0-9\-]', '', corrected_text.strip())
    print("CLEANED TEXT:", cleaned_text)

    if '-' in cleaned_text:
        # Split by the dash
        left, right = cleaned_text.split('-', 1)
        
        try:
            # Get the y-coordinates of the text boxes to determine top vs bottom
            top_box_y = result[0][1][0][1]  # Top text box y-coordinate (1st element)
            bottom_box_y = result[1][1][0][1]  # Bottom text box y-coordinate (2nd element)

            # Assign top score to player 1, bottom score to player 2
            if top_box_y < bottom_box_y:
                player1_score = int(left)
                player2_score = int(right)
            else:
                player1_score = int(right)
                player2_score = int(left)
        except ValueError:
            player1_score = player2_score = None
    else:
        player1_score = player2_score = None

    print("SCORES:", player1_score, player2_score)
    return player1_score, player2_score

def preprocess_scores(scores):
    cleaned_scores = {}
    prev_score = None

    for frame, (score1, score2) in scores.items():
        score1 = score1.strip("-.") if isinstance(score1, str) and score1.strip("-.").isdigit() else score1
        score2 = score2.strip("-.") if isinstance(score2, str) and score2.strip("-.").isdigit() else score2

        if not score1 or not score2:
            continue

        try:
            score1 = int(score1) if score1 != '' else None
            score2 = int(score2) if score2 != '' else None
        except ValueError:
            continue

        current_score = (score1, score2)
        if current_score != prev_score:
            cleaned_scores[frame] = current_score
            prev_score = current_score

    return cleaned_scores

def analyze_scoreboard(video_frames):
    global score_box_coords
    scores = {}
    prev_score = None

    for i, frame in enumerate(video_frames):
        print(f"Scoreboard: {i+1}/{len(video_frames)}")
        score = detect_score(frame)
        if score and score != prev_score:
            scores[i] = score
            prev_score = score

    return scores
