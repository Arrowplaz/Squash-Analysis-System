import pytesseract
import cv2

score_box_coords = []

def get_user_selected_roi(frame, points):

    global score_box_coords
    score_box_coords = points
    return
    print("Select the region of interest (ROI) for the scoreboard.")
    cv2.namedWindow("Select Scoreboard ROI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Select Scoreboard ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    score_box_coords = cv2.selectROI("Select Scoreboard ROI", frame, fromCenter=False, showCrosshair=True)
    
    cv2.destroyAllWindows()
    return score_box_coords

# Function to detect the score from the scoreboard
def detect_score(frame):
    """
    Detects the scores from the scoreboard ROI.
    Assumes the scoreboard has two separate numbers representing player scores.
    """
    global score_box_coords
    x, y, w, h = score_box_coords
    score_roi = frame[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)

    # Determine split point (assuming horizontal split for two numbers)
    split_x = w // 2

    #Split into 2 sub ROIS
    if h > w:
        # Taller than wide: split horizontally
        split_y = h // 2
        player1_roi = gray[:split_y, :]  # Top half
        player2_roi = gray[split_y:, :]  # Bottom half
    else:
        # Wider than tall: split vertically
        split_x = w // 2
        player1_roi = gray[:, :split_x]  # Left half
        player2_roi = gray[:, split_x:]  # Right half

    # Preprocess both sub-ROIs (e.g., thresholding)
    _, player1_thresh = cv2.threshold(player1_roi, 127, 255, cv2.THRESH_BINARY_INV)
    _, player2_thresh = cv2.threshold(player2_roi, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply Tesseract to each sub-ROI
    player1_score = pytesseract.image_to_string(player1_thresh, config='--psm 7 digits').strip()
    player2_score = pytesseract.image_to_string(player2_thresh, config='--psm 7 digits').strip()

    return player1_score, player2_score

def preprocess_scores(scores):
    """
    Preprocess detected scores to clean and normalize the data.
    """
    cleaned_scores = {}
    prev_score = None

    for frame, (score1, score2) in scores.items():
        # Normalize scores
        score1 = score1.strip("-.") if score1.strip("-.").isdigit() else ""
        score2 = score2.strip("-.") if score2.strip("-.").isdigit() else ""

        # Skip if either scores are empty
        if not score1 or not score2:
            continue

        # Convert to integers if valid
        try:
            score1 = int(score1) if score1 else None
            score2 = int(score2) if score2 else None
        except ValueError:
            continue  # Skip invalid scores

        # Check for score changes
        current_score = (score1, score2)
        if current_score != prev_score:
            cleaned_scores[frame] = current_score
            prev_score = current_score

    return cleaned_scores


# Function to analyze scoreboard and segment video by points
def analyze_scoreboard(video_frames):
    global score_box_coords
    scores = {}
    prev_score = None
    counter = 0
    for i, frame in enumerate(video_frames):
        counter += 1
        print(f"Scoreboard: {counter}/{len(video_frames)}")
        score = detect_score(frame)
        if score and score != prev_score:
            scores[i] = score  # Log frame number and score
            prev_score = score
    
    return scores
