import cv2
import inference
from PIL import Image
import numpy as np
from utils import ultralytics_key
class CourtDetector:

    def __init__(self):
        self.model = inference.get_model("court-uzfum-clujb/2", api_key='ultralytics_key', onnxruntime_execution_providers=["CPUExecutionProvider"])

    def detect_court(self, frame, read_from_stub=False, stub_path=None):
        # Inference step
        results = self.model.infer(image=frame)

        print(results)

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
            print(bbox)

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

            # Draw the polygon
            if points and bbox['class_name'] == 'floor':
                points = np.array(points, dtype=np.int32)

                # Draw the blue polygon
                cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)

                # Find the corners (assuming it's a rectangle or quadrilateral)
                rect = cv2.boundingRect(points)  # Get bounding rectangle
                x, y, w, h = rect
                box_points = np.array([
                    (x, y),         # Top-left
                    (x + w, y),     # Top-right
                    (x + w, y + h), # Bottom-right
                    (x, y + h)      # Bottom-left
                ])

                # Draw red circles at the corners
                for (cx, cy) in box_points:
                    cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)  # Red circles

        # Show the frame with bounding boxes, labels, and polygons
        cv2.imshow("Detected Court", frame)
        cv2.waitKey(0)  # Wait indefinitely for a key press
        cv2.destroyAllWindows()

        return []
