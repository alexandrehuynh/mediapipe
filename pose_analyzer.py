import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import pandas as pd
import matplotlib.pyplot as plt

class PoseAnalyzer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.config = self.load_config()
        self.angles_data = []
        self.video_path = None

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

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def put_text_with_background(self, img, text, position, font, font_scale, text_color, thickness):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        bg_rect_x1 = position[0]
        bg_rect_y1 = position[1] - text_size[1] - 10
        bg_rect_x2 = position[0] + text_size[0] + 10
        bg_rect_y2 = position[1]
        
        cv2.rectangle(img, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), (0, 0, 0), -1)
        
        cv2.putText(img, text, (position[0] + 5, position[1] - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def process_video(self, video_path, output_path):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        with self.mp_pose.Pose(
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    self.draw_pose_annotations(image, results.pose_landmarks, frame_width, frame_height)

                out.write(image)
                cv2.imshow('MediaPipe Pose', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def draw_pose_annotations(self, image, pose_landmarks, frame_width, frame_height):
        self.mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        landmarks = pose_landmarks.landmark
        angles = self.calculate_angles(landmarks)
        self.angles_data.append(angles)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_color = (255, 255, 255)

        for angle_name in self.config['angles_to_display']:
            if angle_name in angles:
                angle = angles[angle_name]
                position = landmarks[getattr(self.mp_pose.PoseLandmark, angle_name)]
                coord = (int(position.x * frame_width), int(position.y * frame_height))
                self.put_text_with_background(image, f"{angle_name}: {angle:.0f}", coord, font, font_scale, text_color, thickness)

    def calculate_angles(self, landmarks):
        angles = {}
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

        for joint, (p1, p2, p3) in joints.items():
            angle = self.calculate_angle(
                (landmarks[p1.value].x, landmarks[p1.value].y),
                (landmarks[p2.value].x, landmarks[p2.value].y),
                (landmarks[p3.value].x, landmarks[p3.value].y)
            )
            angles[joint] = angle

        return angles

    def analyze_data(self, selected_angles=None):
        if not self.angles_data:
            print("No data to analyze. Please process a video first.")
            return

        df = pd.DataFrame(self.angles_data)
        
        if selected_angles:
            df = df[selected_angles]

        # Calculate statistics
        stats = df.agg(['mean', 'min', 'max', 'std'])
        print("Angle Statistics:")
        print(stats)

        # Save data
        os.makedirs("angle_data", exist_ok=True)
        base_name = os.path.basename(self.video_path)
        name_without_extension = os.path.splitext(base_name)[0]
        
        # Save raw data
        df.to_csv(f"angle_data/{name_without_extension}_raw_data.csv")
        
        # Save statistics
        stats.to_csv(f"angle_data/{name_without_extension}_statistics.csv")

        # Save statistics as text file for easy reading
        with open(f"angle_data/{name_without_extension}_statistics.txt", 'w') as f:
            f.write("Angle Statistics:\n")
            f.write(stats.to_string())

        # Plot angle variations
        plt.figure(figsize=(12, 6))
        for column in df.columns:
            plt.plot(df[column].rolling(window=5).mean(), label=column)
        plt.legend()
        plt.title("Joint Angle Variations Over Time (5-frame moving average)")
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.savefig(f"angle_data/{name_without_extension}_angle_plot.png")
        plt.show()

class PoseAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pose Analyzer")
        self.pose_analyzer = PoseAnalyzer()

        self.create_widgets()

    def create_widgets(self):
        self.select_video_button = tk.Button(self.master, text="Select Video", command=self.select_video)
        self.select_video_button.pack()

        self.process_video_button = tk.Button(self.master, text="Process Video", command=self.process_video)
        self.process_video_button.pack()

        self.analyze_data_button = tk.Button(self.master, text="Analyze Data", command=self.analyze_data)
        self.analyze_data_button.pack()

        self.angle_vars = {}
        for angle in self.pose_analyzer.config['angles_to_display']:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.master, text=angle, variable=var).pack()
            self.angle_vars[angle] = var

    def select_video(self):
        self.video_path = filedialog.askopenfilename(initialdir="./input", 
                                                     title="Select video file",
                                                     filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if self.video_path:
            messagebox.showinfo("File Selected", f"Selected video: {self.video_path}")

    def process_video(self):
        if hasattr(self, 'video_path'):
            output_path = self.get_output_path(self.video_path)
            self.pose_analyzer.process_video(self.video_path, output_path)
            messagebox.showinfo("Processing Complete", f"Output saved to: {output_path}")
        else:
            messagebox.showerror("Error", "Please select a video first")

    def analyze_data(self):
        selected_angles = [angle for angle, var in self.angle_vars.items() if var.get()]
        self.pose_analyzer.analyze_data(selected_angles)

    def get_output_path(self, input_path):
        base_name = os.path.basename(input_path)
        output_name = f"mapped_{base_name}"
        output_path = os.path.join("output", output_name)
        os.makedirs("output", exist_ok=True)
        return output_path

def main():
    root = tk.Tk()
    app = PoseAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()