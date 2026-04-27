import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter

# Research Safety: Prevents mouse from getting stuck
pyautogui.FAILSAFE = True 

class GazeResearchSystem:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        # Calibration Setup
        self.calibration_points = [(100, 100), (1820, 100), (960, 540), (100, 980), (1820, 980)]
        self.current_point_idx = 0
        self.calibration_data, self.target_data = [], []
        self.is_calibrated = False
        self.regressor = LinearRegression()

        # Research Features
        self.gaze_history = []  # For Heatmap
        self.smoother_x, self.smoother_y = 0, 0
        self.alpha = 0.2  # Smoothing factor (0.1 = slow/stable, 0.5 = fast/jittery)

    def get_iris_center(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0]
            # Precise iris centers (468 = Left, 473 = Right)
            lx, ly = mesh.landmark[468].x, mesh.landmark[468].y
            rx, ry = mesh.landmark[473].x, mesh.landmark[473].y
            return np.array([(lx + rx) / 2, (ly + ry) / 2])
        return None

    def generate_heatmap(self, shape):
        """Creates a density-based heatmap from gaze history"""
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        # Only use the last 400 points for a 'rolling' heatmap
        for (x, y) in self.gaze_history[-400:]:
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                heatmap[int(y), int(x)] += 1
        
        # Apply Gaussian blur to create the 'heat' effect
        heatmap = gaussian_filter(heatmap, sigma=15)
        if np.max(heatmap) > 0:
            heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return color_heatmap

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Research", cv2.WND_PROP_FULLSCREEN)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            iris_pos = self.get_iris_center(frame)

            if not self.is_calibrated:
                # --- CALIBRATION MODE ---
                target = self.calibration_points[self.current_point_idx]
                cv2.circle(canvas, target, 20, (0, 255, 0), -1)
                if cv2.waitKey(1) & 0xFF == ord('c') and iris_pos is not None:
                    self.calibration_data.append(iris_pos)
                    self.target_data.append(target)
                    self.current_point_idx += 1
                    if self.current_point_idx >= len(self.calibration_points):
                        self.regressor.fit(self.calibration_data, self.target_data)
                        self.is_calibrated = True
            else:
                # --- RESEARCH MODE (Accessibility & Heatmap) ---
                if iris_pos is not None:
                    pred = self.regressor.predict([iris_pos])[0]
                    px, py = int(pred[0]), int(pred[1])

                    # 1. Accessibility: Exponential Moving Average Smoothing
                    self.smoother_x = (self.alpha * px) + (1 - self.alpha) * self.smoother_x
                    self.smoother_y = (self.alpha * py) + (1 - self.alpha) * self.smoother_y
                    pyautogui.moveTo(int(self.smoother_x), int(self.smoother_y))

                    # 2. Heatmapping: Collect and overlay
                    self.gaze_history.append((px, py))
                    heatmap_img = self.generate_heatmap(canvas.shape)
                    canvas = cv2.addWeighted(canvas, 0.3, heatmap_img, 0.7, 0)

                    # Visual cursor for feedback
                    cv2.circle(canvas, (px, py), 10, (255, 255, 255), 2)

            cv2.imshow("Research", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    GazeResearchSystem().run()
