import cv2
court_keypoints = []






# Mouse callback function to capture points
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the point when the left mouse button is clicked
        court_keypoints.append((x, y))
        print(f"Point {len(court_keypoints)} selected: ({x}, {y})")

        cv2.circle(param['frame'], (x, y), 5, (0, 0, 255), -1)  # Draw red dot
        cv2.imshow("Select Court Points", param['frame'])

def get_user_selected_points(frame):
    global court_keypoints
    court_keypoints = []  # Clear previous points
    cv2.namedWindow("Select Court Points")
    
    # Pass frame as part of the param dictionary
    cv2.setMouseCallback("Select Court Points", select_points, param={'frame': frame})

    print("Select 4 points in the order: top-left, top-right, bottom-left, bottom-right.")
    
    # Display the frame and wait for the user to select points
    while len(court_keypoints) < 4:
        cv2.imshow("Select Court Points", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
            break

    cv2.destroyWindow("Select Court Points")
    return [point for sublist in court_keypoints for point in sublist]  # Flatten list