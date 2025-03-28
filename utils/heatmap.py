import cv2
import numpy as np

def add_overlay(heatmap, height, width, padding=50):
    """
    Draws court lines and service boxes on a heatmap overlay.
    
    Parameters:
        heatmap (np.array): The canvas to draw on.
        height (int): Height of the court area.
        width (int): Width of the court area.
        padding (int): Padding around the court.
        
    Returns:
        heatmap (np.array): Canvas with the drawn overlay.
    """
    # Define the four padded corners
    top_left = (0, 0)
    top_right = (width + padding, padding)
    bottom_left = (0, height + padding)
    bottom_right = (width + padding, height + padding)
    
    # Draw the outer rectangle
    cv2.rectangle(heatmap, top_left, bottom_right, (50, 50, 50), 2)
    
    # Calculate dimensions
    length_in_pixels = bottom_left[1] - top_left[1]
    width_in_pixels = top_right[0] - top_left[0]
    
    # Draw short line (using real-world ratio: 5.44 / 9.75)
    short_line_ratio = 5.44 / 9.75
    short_line_draw_y = int((length_in_pixels * short_line_ratio) + top_left[1])
    cv2.line(heatmap, (top_left[0], short_line_draw_y), (top_right[0], short_line_draw_y), (50, 50, 50), 2)
    
    # Draw mid-court line
    mid_court_line_draw_x = int((width_in_pixels * 0.5) + top_left[0])
    cv2.line(heatmap, (mid_court_line_draw_x, bottom_left[1]), (mid_court_line_draw_x, short_line_draw_y), (50, 50, 50), 2)
    
    # Service box dimensions in meters (converted to pixels)
    service_box_length_m = 1.6
    service_box_width_m = 1.6
    service_box_pixel_length = int(length_in_pixels * (service_box_length_m / 9.75))
    service_box_pixel_width = int(width_in_pixels * (service_box_width_m / 6.4))
    
    # Left service box (from top-left)
    box1_top_left = (top_left[0], short_line_draw_y)
    box1_bottom_right = (top_left[0] + service_box_pixel_width, short_line_draw_y + service_box_pixel_length)
    cv2.rectangle(heatmap, box1_top_left, box1_bottom_right, (50, 50, 50), 2)
    
    # Right service box (from top-right)
    box2_top_left = (top_right[0] - service_box_pixel_width, short_line_draw_y)
    box2_bottom_right = (top_right[0], short_line_draw_y + service_box_pixel_length)
    cv2.rectangle(heatmap, box2_top_left, box2_bottom_right, (50, 50, 50), 2)
    
    return heatmap

def create_heatmap(frame, court_keypoints, overlay_width=1800):
    """
    Computes the homography to map the input squash court image to a standardized overlay,
    and returns the warped image, the overlay (with court lines), and the homography matrix.
    
    Parameters:
        image_path (str): Path to the input image.
        court_keypoints (list): Four (x, y) points (top-left, top-right, bottom-left, bottom-right) in the image.
        overlay_width (int): Desired width (in pixels) of the overlay.
        
    Returns:
        warped_image (np.array): The warped image of the court.
        overlay (np.array): The overlay canvas with drawn court lines.
        homography_matrix (np.array): The computed homography matrix.
    """
    
    # Use standard squash court dimensions (9.75m x 6.4m) for aspect ratio
    aspect_ratio = 9.75 / 6.4  # length/width
    overlay_height = int(overlay_width * aspect_ratio)
    
    # Define destination points for the overlay (order: top-left, top-right, bottom-left, bottom-right)

    overlay_corners = np.array([
        [0, 0],
        [overlay_width, 0],
        [0, (overlay_height * 7.04) / 9.75], 
        [overlay_width, (overlay_height * 7.04) / 9.75]
    ], dtype=np.float32)
    
    # court_keypoints = [(951, 754), (1952, 754), (714, 1344), (2176, 1342), (598, 1576), (2271, 1575)]
    # Convert court keypoints to NumPy array and compute the homography matrix
    src_points = np.array(court_keypoints, dtype=np.float32)
    homography_matrix, status = cv2.findHomography(src_points, overlay_corners)
    if homography_matrix is None:
        raise ValueError("Homography matrix computation failed.")
    
    # Warp the original image using the computed homography
    warped_image = cv2.warpPerspective(frame, homography_matrix, (overlay_width, overlay_height))
    
    # Create an overlay canvas with padding (for drawing)
    padding = 50
    overlay_canvas = np.zeros((overlay_height + padding, overlay_width + padding, 3), dtype=np.uint8)
    overlay_canvas = add_overlay(overlay_canvas, overlay_height, overlay_width, padding=padding)
    
    return warped_image, overlay_canvas, homography_matrix

def map_detections(detections, homography_matrix):
    # Initialize an empty list to store the coordinates of interest
    points = []
    # Iterate over the list of dictionaries in detections
    for detection in detections:
        # Assuming each dictionary contains two bboxes, one for each player
        for player in detection.values():
            # Extract the x, y, width, height from the bbox dictionary for each player
            # Adjust these keys if your dictionary format is different
            x1, y1, x2, y2 = player
            center_x = int((x1 + x2) / 2)
            center_y = y2
            
            
            # Append the center as a tuple to the points list
            points.append((center_x, center_y))

    # Convert points into a numpy array of floats
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    # Apply the homography transformation using the provided matrix
    mapped_points = cv2.perspectiveTransform(points, homography_matrix)

    # Convert back to a list of (x, y) tuples
    return mapped_points.reshape(-1, 2)



def overlay_heatmap(composite, mapped_detections):
    heatmap = np.zeros((composite.shape[0], composite.shape[1]), dtype=np.float32)

    # Accumulate intensity at detected points
    for (x, y) in mapped_detections:
        x, y = int(x), int(y)  # Ensure coordinates are integers
        cv2.circle(heatmap, (x, y), radius=20, color=255, thickness=-1)  # Larger radius for spread

    # Apply Gaussian blur for a smoother heatmap
    heatmap = cv2.GaussianBlur(heatmap, (35, 35), 0) #Potentially make the kernal larger

    # Normalize heatmap to range 0-255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 3-channel color heatmap using a colormap
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend the heatmap with the composite image
    alpha = 0.6  # Transparency factor
    heatmap_overlay = cv2.addWeighted(heatmap_color, alpha, composite, 1 - alpha, 0)

    # # Display the final result
    # cv2.setWindowProperty("Court with Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Court with Heatmap", heatmap_overlay)
    cv2.waitKey(0)

    return heatmap

