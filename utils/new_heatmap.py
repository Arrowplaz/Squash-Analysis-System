import cv2
import numpy as np

def create_overlay(scale_height, scale_width, padding=50):
    """Create a squash court overlay as a transparent layer."""
    court_length_m, court_width_m = 9.75, 6.4
    height, width = int(court_length_m * scale_height), int(court_width_m * scale_width)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    top_left = (padding, padding)
    top_right = (width - padding, padding)
    bottom_left = (padding, height - padding)
    bottom_right = (width - padding, height - padding)
    
    # Court rectangle
    cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0, 255), 2)

    length_in_pixels = bottom_left[1] - top_left[1]
    width_in_pixels = top_right[0] - top_left[0]

    # Short line
    short_line_ratio = 5.44 / 9.75
    short_line_draw_y = (length_in_pixels * short_line_ratio) + top_left[1]
    cv2.line(
        overlay, 
        (top_left[0], int(short_line_draw_y)), 
        (top_right[0], int(short_line_draw_y)), 
        (0, 255, 0, 255), 2
    )

    # Mid-court line
    mid_court_line_ratio = 0.5
    mid_court_line_draw_x = (width_in_pixels * mid_court_line_ratio) + bottom_left[0]
    cv2.line(
        overlay, 
        (int(mid_court_line_draw_x), bottom_left[1]), 
        (int(mid_court_line_draw_x), int(short_line_draw_y)), 
        (0, 255, 0, 255), 2
    )

    # Service boxes
    service_box_length_m = 1.6  
    service_box_width_m = 1.6  
    service_box_ratio_length = service_box_length_m / 9.75
    service_box_ratio_width = service_box_width_m / 6.4
    service_box_pixel_length = length_in_pixels * service_box_ratio_length
    service_box_pixel_width = width_in_pixels * service_box_ratio_width

    # Left service box
    top_left_left_box = (top_left[0], int(short_line_draw_y))
    bottom_right_left_box = (
        top_left[0] + int(service_box_pixel_width),
        int(short_line_draw_y) + int(service_box_pixel_length)
    )
    cv2.rectangle(overlay, top_left_left_box, bottom_right_left_box, (0, 255, 0, 255), 2)

    # Right service box
    top_left_right_box = (
        top_right[0] - int(service_box_pixel_width),
        int(short_line_draw_y)
    )
    bottom_right_right_box = (
        top_right[0],
        int(short_line_draw_y) + int(service_box_pixel_length)
    )
    cv2.rectangle(overlay, top_left_right_box, bottom_right_right_box, (0, 255, 0, 255), 2)

    return overlay, [top_left, top_right, bottom_left, bottom_right]

def perform_perspective_transform(image, src_points, dst_points):
    """Perform a perspective transform from src_points to dst_points."""
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Calculate the output dimensions based on the overlay size
    dsize = (int(dst_points[2][0]), int(dst_points[2][1]))  # Convert to integers
    warped_image = cv2.warpPerspective(image, matrix, dsize)

    return warped_image, matrix

def overlay_court_on_match(original_image, court_keypoints):
    """Overlay the squash court lines on the original match footage."""
    scale_height = original_image.shape[0] / 9.75
    scale_width = original_image.shape[1] / 6.4

    # Create overlay and get overlay corners
    overlay, overlay_corners = create_overlay(scale_height, scale_width)

    # Perform perspective transform on the overlay to match the original image
    warped_overlay, _ = perform_perspective_transform(overlay, overlay_corners, court_keypoints)

    # Create a mask from the warped overlay
    mask = warped_overlay[:, :, 3] / 255.0
    mask = cv2.merge([mask, mask, mask])

    # Blend the overlay with the original image
    combined = original_image.copy()
    combined = (combined * (1 - mask) + warped_overlay[:, :, :3] * mask).astype(np.uint8)

    return combined

# Example usage
original_image = cv2.imread("/Users/anagireddygari/Desktop/Honors Project/Honors-Project-Player-Tracking-in-Squash-for-Analytics/input_videos/veer_ss.png")  # Load original squash match frame

# Example detected court corners from video (top-left, top-right, bottom-left, bottom-right)
court_keypoints = [(951, 754), (1952,  754), (623, 1579), (2275, 1579)]

# Overlay the court on the match footage
result_image = overlay_court_on_match(original_image, court_keypoints)

# Display the result
cv2.imshow('Squash Court with Overlay', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()