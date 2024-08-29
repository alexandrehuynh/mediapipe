import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class PoseAnalyzer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.config = self.load_config()
        self.angles_data = []
        self.file_path = None
        self.detected_joints = set()

    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.7,
                'enable_gpu': False,
                'process_every_n_frames': 1,
                'angles_to_display': [
                    'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE',
                    'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'
                ]
            }

    def configure_mediapipe(self):
        return self.mp_pose.Pose(
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360 - angle if angle > 180.0 else angle

    def get_landmark_coords(self, landmark):
        return [landmark.x, landmark.y]

    def get_output_path(self, input_path, prefix="mapped_"):
        base_name = os.path.basename(input_path)
        output_name = f"{prefix}{base_name}"
        output_path = os.path.join("output", output_name)
        os.makedirs("output", exist_ok=True)
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        return output_path

    def process_file(self, file_path, output_path, joints_to_process):
        self.file_path = file_path
        self.angles_data = []
        self.detected_joints.clear()

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self.process_image(file_path, output_path, joints_to_process)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            return self.process_video(file_path, output_path, joints_to_process)
        else:
            return False, "Unsupported file format"

    def process_video(self, video_path, output_path, joints_to_process):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Error: Unable to open video file {video_path}"

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        with self.configure_mediapipe() as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % self.config['process_every_n_frames'] != 0:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    self.draw_pose_annotations(image, results, frame_width, frame_height, joints_to_process)
                    angles = self.calculate_angles(results, joints_to_process)
                    self.angles_data.append(angles)
                    self.detected_joints.update(angles.keys())

                out.write(image)

        cap.release()
        out.release()

        if frame_count == 0:
            return False, f"Error: No frames were processed from {video_path}"
        else:
            return True, f"Successfully processed {frame_count} frames from {video_path}"

    def process_image(self, image_path, output_path, joints_to_process):
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Error: Unable to read image file {image_path}"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.configure_mediapipe() as pose:
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                self.draw_pose_annotations(image, results, image.shape[1], image.shape[0], joints_to_process)
                angles = self.calculate_angles(results, joints_to_process)
                self.angles_data.append(angles)
                self.detected_joints.update(angles.keys())
                cv2.imwrite(output_path, image)
                return True, f"Successfully processed image and saved to {output_path}"
            else:
                return False, "No pose landmarks detected in the image"

    def draw_pose_annotations(self, image, results, frame_width, frame_height, joints_to_process):
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_landmarks.landmark
        angles = self.calculate_angles(results, joints_to_process)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_color = (255, 255, 255)

        for angle_name in joints_to_process:
            if angle_name in angles:
                angle_info = angles[angle_name]
                position = landmarks[getattr(self.mp_pose.PoseLandmark, angle_name)]
                coord = (int(position.x * frame_width), int(position.y * frame_height))

                if angle_info['angle'] is not None:
                    text = f"{angle_name}: {angle_info['angle']:.0f}"
                else:
                    text = f"{angle_name}: N/A"

                if 'confidence' in angle_info:
                    text += f" (conf: {angle_info['confidence']:.2f})"
                self.put_text_with_background(image, text, coord, font, font_scale, text_color, thickness)

        self.draw_spine_curve(image, landmarks)
        
    def put_text_with_background(self, img, text, position, font, font_scale, text_color, thickness):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        bg_rect_x1 = position[0]
        bg_rect_y1 = position[1] - text_size[1] - 10
        bg_rect_x2 = position[0] + text_size[0] + 10
        bg_rect_y2 = position[1]
        cv2.rectangle(img, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), (0, 0, 0), -1)
        cv2.putText(img, text, (position[0] + 5, position[1] - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def draw_spine_curve(self, image, landmarks):
        # Get relevant keypoints
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate midpoints
        mid_shoulder = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
        mid_hip = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])

        # Create spine points
        spine_points = np.array([
            [nose.x, nose.y],
            mid_shoulder,
            (mid_shoulder + mid_hip) / 2,  # Add a point between shoulder and hip
            mid_hip
        ])

        # Remove any NaN values
        spine_points = spine_points[~np.isnan(spine_points).any(axis=1)]

        if len(spine_points) < 3:
            print("Not enough valid points to create a spine curve")
            return

        # Create a smooth curve using CubicSpline
        t = np.linspace(0, 1, len(spine_points))
        cs = CubicSpline(t, spine_points, bc_type='natural')
        t_smooth = np.linspace(0, 1, 50)  # Reduce number of points
        smooth_spine = cs(t_smooth)

        # Convert to pixel coordinates
        h, w = image.shape[:2]
        smooth_spine_px = np.int32(smooth_spine * [w, h])

        # Draw the main curve
        cv2.polylines(image, [smooth_spine_px], False, (0, 255, 0), 2)

        # Draw circles for major joints
        for point in spine_points:
            cv2.circle(image, tuple(np.int32(point * [w, h])), 5, (0, 200, 0), -1)

        # Add labels
        labels = ["C1", "T1", "L1", "S1"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_positions = [0, 0.3, 0.7, 1]  # Adjusted positions along the spine
        for label, pos in zip(labels, label_positions):
            index = min(int(pos * (len(smooth_spine_px) - 1)), len(smooth_spine_px) - 1)
            point = smooth_spine_px[index]
            cv2.putText(image, label, tuple(point), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    def calculate_angles(self, results, joints_to_process):
        angles = {}
        landmarks = results.pose_world_landmarks if self.mode == '3d' and results.pose_world_landmarks else results.pose_landmarks
        if not landmarks:
            return angles

        joints = {
            'LEFT_SHOULDER': (self.mp_pose.PoseLandmark.LEFT_EAR, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            'LEFT_ELBOW': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            'LEFT_WRIST': (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_INDEX),
            'LEFT_HIP': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            'LEFT_KNEE': (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            'LEFT_ANKLE': (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
            'RIGHT_SHOULDER': (self.mp_pose.PoseLandmark.RIGHT_EAR, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            'RIGHT_ELBOW': (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'RIGHT_WRIST': (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_INDEX),
            'RIGHT_HIP': (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            'RIGHT_KNEE': (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            'RIGHT_ANKLE': (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL)
        }

        for joint in joints_to_process:
            if joint in joints:
                p1, p2, p3 = joints[joint]
                try:
                    coords1 = self.get_landmark_coords(landmarks.landmark[p1.value])
                    coords2 = self.get_landmark_coords(landmarks.landmark[p2.value])
                    coords3 = self.get_landmark_coords(landmarks.landmark[p3.value])
                    
                    visibility = min(landmarks.landmark[p.value].visibility for p in (p1, p2, p3))
                    
                    if visibility > 0.5:
                        angle = self.calculate_angle(coords1, coords2, coords3)
                        angles[joint] = {
                            'angle': angle,
                            'confidence': visibility
                        }
                    else:
                        angles[joint] = {
                            'angle': None,
                            'confidence': visibility,
                            'error': 'Low visibility'
                        }
                except Exception as e:
                    angles[joint] = {
                        'angle': None,
                        'confidence': 0,
                        'error': str(e)
                    }

        return angles

    def analyze_data(self, selected_angles=None):
        if not self.angles_data:
            return False, "No data to analyze. Please process a file first."

        df = pd.DataFrame(self.angles_data)
        detected_columns = [col for col in df.columns if col in self.detected_joints]
        df = df[detected_columns]

        if selected_angles:
            df = df[[col for col in selected_angles if col in detected_columns]]

        angle_df = df.applymap(lambda x: x['angle'] if isinstance(x, dict) and 'angle' in x else None)
        confidence_df = df.applymap(lambda x: x['confidence'] if isinstance(x, dict) and 'confidence' in x else None)

        stats = angle_df.where(confidence_df > 0.5).agg(['mean', 'min', 'max', 'std'])

        os.makedirs("angle_data", exist_ok=True)
        base_name = os.path.basename(self.file_path)
        name_without_extension = os.path.splitext(base_name)[0]
        mode_prefix = f"{self.mode.upper()}_"

        df.to_csv(f"angle_data/{mode_prefix}{name_without_extension}_raw_data.csv")
        stats.to_csv(f"angle_data/{mode_prefix}{name_without_extension}_statistics.csv")

        with open(f"angle_data/{mode_prefix}{name_without_extension}_statistics.txt", 'w') as f:
            f.write(f"Angle Statistics ({self.mode.upper()} mode):\n")
            f.write(stats.to_string())

        try:
            plt.figure(figsize=(12, 6))
            for column in angle_df.columns:
                plt.plot(angle_df[column].rolling(window=5).mean(), label=column)
            plt.legend()
            plt.title(f"Detected Joint Angle Variations Over Time ({self.mode.upper()} mode, 5-frame moving average)")
            plt.xlabel("Frame")
            plt.ylabel("Angle (degrees)")
            plt.savefig(f"angle_data/{mode_prefix}{name_without_extension}_angle_plot.png")
            plt.close()
        except Exception as e:
            print(f"Error creating plot: {e}")
            return False, f"Analysis complete, but there was an error creating the plot: {e}"

        return True, f"Analysis complete. Results saved in angle_data/{mode_prefix}{name_without_extension}_*.csv/txt/png"

