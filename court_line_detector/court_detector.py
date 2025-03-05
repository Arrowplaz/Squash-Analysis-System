import cv2
import inference
import numpy as np
from utils import ultralytics_key
from scipy.spatial import ConvexHull

class CourtDetector:
    def __init__(self):
        self.model = inference.get_model("court-uzfum-clujb/2", api_key='AhGS5Qpq2TRxwQEakFeH', onnxruntime_execution_providers=["CPUExecutionProvider"])

    def detect_court(self, frame, read_from_stub=False, stub_path=None, show = True):
        results = self.model.infer(image=frame)

        for prediction in results[0].predictions:

            points = np.array([(int(p.x), int(p.y)) for p in prediction.points], dtype=np.int32)
            if len(points) < 3:
                continue  # Ignore invalid detections

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            center = np.mean(hull_points, axis=0)
            hull_points = sorted(hull_points, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))

            # Identify corners
            top_left = min(hull_points, key=lambda p: p[0] + p[1])
            top_right = max(hull_points, key=lambda p: p[0] - p[1])
            bottom_right = max(hull_points, key=lambda p: p[0] + p[1])
            bottom_left = min(hull_points, key=lambda p: p[0] - p[1])

            corners = [top_left, top_right, bottom_right, bottom_left]

            if show:
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                for i in range(4):
                    cv2.line(frame, corners[i], corners[(i+1) % 4], colors[i], thickness=2)

                for corner in corners:
                    cv2.circle(frame, corner, radius=5, color=(0, 0, 255), thickness=-1)

                cv2.putText(frame, prediction.class_name, (corners[0][0], corners[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.imshow("Detected Court", frame)
                cv2.waitKey(0)

            return corners
        return None
