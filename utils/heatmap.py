import cv2
import numpy as np

court_width_m, court_height_m = 6.4, 9.75

# Resolution (pixels per meter)
pixels_per_meter = 100

# Court dimensions in pixels
heatmap_width, heatmap_height = int(court_width_m * pixels_per_meter), int(court_height_m * pixels_per_meter)

def create_court_heatmap(player_detections, court_keypoints, heatmap_size=(500, 300), num_players=2):
    # Initialize an empty heatmap for each player
    heatmaps = [np.zeros(heatmap_size, dtype=np.float32) for _ in range(num_players)]
    
    # Get the transformation matrix from the court points to the heatmap space
    src_pts = np.array([
        [court_keypoints[0], court_keypoints[1]],
        [court_keypoints[2], court_keypoints[3]],
        [court_keypoints[4], court_keypoints[5]],
        [court_keypoints[6], court_keypoints[7]]
    ], dtype=np.float32)
    
    dest_pts = np.array([
        [0, 0],
        [heatmap_size[1] - 1, 0],
        [0, heatmap_size[0] - 1],
        [heatmap_size[1] - 1, heatmap_size[0] - 1]
    ], dtype=np.float32)
    
    transformation_matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)

    # Accumulate player positions on the heatmap
    for frame_detections in player_detections:
        for player_id, bbox in frame_detections.items():
            # Calculate the center of the bounding box
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            point = np.array([[x_center, y_center]], dtype=np.float32)
            
            # Transform the point to heatmap coordinates
            transformed_point = cv2.perspectiveTransform(np.array([point]), transformation_matrix)[0][0]
            x, y = int(transformed_point[0]), int(transformed_point[1])
            
            # Accumulate intensity on the heatmap for this player
            print(player_id)
            if player_id == 4:
                player_id = 2
            if 0 <= x < heatmap_size[1] and 0 <= y < heatmap_size[0]:
                heatmaps[player_id - 1][y, x] += 1
    
    blurred_heatmaps = []
    for heatmap in heatmaps:
        blurred_heatmap = cv2.GaussianBlur(heatmap, (21, 21), sigmaX=0, sigmaY=0)  # Blurs around points
        normalized_heatmap = cv2.normalize(blurred_heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        blurred_heatmaps.append(colored_heatmap)

    return blurred_heatmaps


def display_heatmaps(heatmaps):
    for i, heatmap in enumerate(heatmaps):
        cv2.imshow(f"Player {i + 1} Heatmap", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


