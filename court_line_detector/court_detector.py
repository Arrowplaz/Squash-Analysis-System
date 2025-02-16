import cv2
import inference
from PIL import Image
import numpy as np
from utils import ultralytics_key
from scipy.spatial import ConvexHull
class CourtDetector:

    def __init__(self):
        self.model = inference.get_model("court-uzfum-clujb/2", api_key='AhGS5Qpq2TRxwQEakFeH', onnxruntime_execution_providers=["CPUExecutionProvider"])

    def detect_court(self, frame, read_from_stub=False, stub_path=None, show = False):
        # Inference step
        results = self.model.infer(image=frame)


        # Draw bounding boxes and labels on the frame
        for prediction in results[0].predictions:
            bbox = {
                'class_name': prediction.class_name,
                'x': prediction.x,
                'y': prediction.y,
                'width': prediction.width,
                'height': prediction.height,
                'points': prediction.points
            }

            # top = int(int(bbox['y']) - (int(bbox['height']) / 2))
            # bottom = int(int(bbox['y']) + (int(bbox['height']) / 2))
            # left = int(int(bbox['x']) - (int(bbox['width']) / 2))
            # right = int(int(bbox['x']) + (int(bbox['width']) / 2))
            # # Calculate bounding box coordinates
            # top_left = (top, left)
            # bottom_right = (bottom, right)

            

            # # Draw rectangle
            # cv2.circle(frame, (int(bbox['x']), int(bbox['y'])), radius=10, color=(0,0,255), thickness=-1)
            # cv2.circle(frame, (int(bbox['x']), int(bbox['y']) + (int(bbox['height']) // 2)), radius=10, color=(0,255,0), thickness=-1)
            # cv2.circle(frame, (int(bbox['x']), int(bbox['y']) - (int(bbox['height']) // 2)), radius=10, color=(0,255,0), thickness=-1)
            # cv2.circle(frame, (int(bbox['x']) + int(bbox['width'] // 2), int(bbox['y'])), radius=10, color=(0,255,0), thickness=-1)
            # cv2.circle(frame, (int(bbox['x']) - int(bbox['width'] // 2), int(bbox['y'])), radius=10, color=(0,255,0), thickness=-1)
            # cv2.putText(frame, bbox['class_name'], (int(bbox['x']) + 5, int(bbox['y']) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            

            # Extract points from the 'points' and draw the polygon
            points = [(int(point.x), int(point.y)) for point in bbox['points']]  # Convert Point(x, y) to tuples
            corners = []
            # Draw the polygon
            if points and bbox['class_name'] == 'floor':
                points = np.array(points, dtype=np.int32)
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                center = np.mean(hull_points, axis=0)
                hull_points = sorted(
                    hull_points,
                    key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0]))
                )
                top_left = min(hull_points, key=lambda p: p[0] + p[1])
                top_right = max(hull_points, key=lambda p: p[0] - p[1])
                bottom_right = max(hull_points, key=lambda p: p[0] + p[1])
                bottom_left = min(hull_points, key=lambda p: p[0] - p[1])

                corners = [top_left, top_right, bottom_left, bottom_right]

                if show:
                    # Draw the blue polygon
                    cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.circle(frame, top_left, radius=10, color=(0,0,255), thickness=-1)
                    cv2.circle(frame, top_right, radius=10, color=(0,0,255), thickness=-1)
                    cv2.circle(frame, bottom_right, radius=10, color=(0,0,255), thickness=-1)
                    cv2.circle(frame, bottom_left, radius=10, color=(0,0,255), thickness=-1)
                    # Show the frame with bounding boxes, labels, and polygons
                    cv2.imshow("Detected Court", frame)
                    cv2.waitKey(0)  # Wait indefinitely for a key press
                    cv2.destroyAllWindows()

        return corners
