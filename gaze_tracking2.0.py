import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LinearRegression

class GazeEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        # Calibration State
        self.calibration_points = [
            (100, 100), (1820, 100), (960, 540), (100, 980), (1820, 980) # 5 Points
        ]
        self.current_point_idx = 0
        self.calibration_data = [] # Stores [iris_x, iris_y]
        self.target_data = []      # Stores [screen_x, screen_y]
        self.is_calibrated = False
        self.regressor = LinearRegression()

    def get_iris_center(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0]
            # Average of left (468) and right (473) iris center landmarks
            lx, ly = mesh.landmark[468].x, mesh.landmark[468].y
            rx, ry = mesh.landmark[473].x, mesh.landmark[473].y
            return np.array([(lx + rx) / 2, (ly + ry) / 2])
        return None

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8) # Fullscreen black canvas
            
            iris_pos = self.get_iris_center(frame)

            if not self.is_calibrated:
                # 1. Calibration Screen
                target = self.calibration_points[self.current_point_idx]
                cv2.circle(canvas, target, 20, (0, 255, 0), -1)
                cv2.putText(canvas, "Look at the green dot and press 'C'", (800, 500), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if cv2.waitKey(1) & 0xFF == ord('c') and iris_pos is not None:
                    # 2. Sample Collection
                    self.calibration_data.append(iris_pos)
                    self.target_data.append(target)
                    self.current_point_idx += 1
                    
                    if self.current_point_idx >= len(self.calibration_points):
                        # 3. Regression Mapping & Model Storage
                        self.regressor.fit(self.calibration_data, self.target_data)
                        self.is_calibrated = True
            else:
                # 4. Prediction Mode
                if iris_pos is not None:
                    prediction = self.regressor.predict([iris_pos])[0]
                    px, py = int(prediction[0]), int(prediction[1])
                    cv2.circle(canvas, (px, py), 15, (0, 0, 255), -1)

            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    GazeEstimator().run()
