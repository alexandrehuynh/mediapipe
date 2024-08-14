import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import CubicSpline

class PoseAnalyzer:
    def __init__(self, mode='3d'):
        # Initialize MediaPipe drawing and pose solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Set the mode (2d or 3d)
        self.mode = mode
        
        # Load configuration settings
        self.config = self.load_config()
        
        # Initialize variables to store angles data and file path
        self.angles_data = []
        self.file_path = None
        
        # Set to keep track of detected joints
        self.detected_joints = set()

    def load_config(self):
        # Load configuration from a JSON file or use default settings if the file is not found
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
        self.mp_pose = mp.solutions.pose  # Reinitialize mp_pose
        if self.mode == '3d':
            return self.mp_pose.Pose(
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence'],
                model_complexity=2,
                enable_segmentation=True,
                smooth_segmentation=True,
                static_image_mode=False
            )
        else:
            return self.mp_pose.Pose(
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        if self.mode == '2d':
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            return 360 - angle if angle > 180.0 else angle
        else:  # 3D mode
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

    def get_landmark_coords(self, landmark):
        if self.mode == '3d':
            return [landmark.x, landmark.y, landmark.z]
        else:
            return [landmark.x, landmark.y]

    def put_text_with_background(self, img, text, position, font, font_scale, text_color, thickness):
        # Calculate the size of the text
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Define the background rectangle coordinates
        bg_rect_x1 = position[0]
        bg_rect_y1 = position[1] - text_size[1] - 10
        bg_rect_x2 = position[0] + text_size[0] + 10
        bg_rect_y2 = position[1]
        
        # Draw the background rectangle
        cv2.rectangle(img, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), (0, 0, 0), -1)
        
        # Draw the text over the background
        cv2.putText(img, text, (position[0] + 5, position[1] - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def process_file(self, file_path, output_path, joints_to_process):
        # Set the file path and clear previous data
        self.file_path = file_path
        self.angles_data = []
        self.detected_joints.clear()

        # Process the file based on its type (image or video)
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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

                    # Process the frame
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
                    cv2.imshow('MediaPipe Pose', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    yield int((frame_count / total_frames) * 100)

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            if frame_count == 0:
                return False, f"Error: No frames were mapped from {video_path}"
            else:
                return True, f"Successfully mapped {frame_count} frames from {video_path}"

    def process_image(self, image_path, output_path, joints_to_process):
            image = cv2.imread(image_path)
            if image is None:
                return False, f"Error: Unable to read image file {image_path}"

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with self.configure_mediapipe() as pose:
                results = pose.process(image_rgb)
                
                if results.pose_landmarks or (self.mode == '3d' and results.pose_world_landmarks):
                    self.draw_pose_annotations(image, results, image.shape[1], image.shape[0], joints_to_process)
                    angles = self.calculate_angles(results, joints_to_process)
                    self.angles_data.append(angles)
                    self.detected_joints.update(angles.keys())
                    cv2.imwrite(output_path, image)
                    return True, f"Successfully processed image and saved to {output_path}"
                else:
                    return False, "No pose landmarks detected in the image"
        
    def draw_pose_annotations(self, image, results, frame_width, frame_height, joints_to_process):
        # Draw the pose landmarks on the image
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_world_landmarks.landmark if self.mode == '3d' else results.pose_landmarks.landmark
        angles = self.calculate_angles(results, joints_to_process)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_color = (255, 255, 255)

        for angle_name in joints_to_process:
            if angle_name in angles:
                angle_info = angles[angle_name]
                position = landmarks[getattr(self.mp_pose.PoseLandmark, angle_name)]
                
                # Calculate coordinates (same for both 2D and 3D)
                coord = (int(position.x * frame_width), int(position.y * frame_height))
                
                # Check if angle is None before formatting
                if angle_info['angle'] is not None:
                    text = f"{angle_name}: {angle_info['angle']:.0f}"
                else:
                    text = f"{angle_name}: N/A"
                
                if 'confidence' in angle_info:
                    text += f" (conf: {angle_info['confidence']:.2f})"
                self.put_text_with_background(image, text, coord, font, font_scale, text_color, thickness)

        self.draw_spine_curve(image, landmarks)
        
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
        
        # Filter out joints that were not detected
        detected_columns = [col for col in df.columns if col in self.detected_joints]
        df = df[detected_columns]

        if selected_angles:
            df = df[[col for col in selected_angles if col in detected_columns]]

        # Extract angles and confidences
        angle_df = df.applymap(lambda x: x['angle'] if isinstance(x, dict) and 'angle' in x else None)
        confidence_df = df.applymap(lambda x: x['confidence'] if isinstance(x, dict) and 'confidence' in x else None)

        # Calculate statistics only for angles with confidence > 0.5
        stats = angle_df.where(confidence_df > 0.5).agg(['mean', 'min', 'max', 'std'])
        
        print("Angle Statistics:")
        print(stats)

        os.makedirs("angle_data", exist_ok=True)
        base_name = os.path.basename(self.file_path)
        name_without_extension = os.path.splitext(base_name)[0]
        
        # Add mode to file names
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

    def get_output_path(self, input_path, prefix="mapped_"):
        base_name = os.path.basename(input_path)
        output_name = f"{prefix}{self.mode.upper()}_{base_name}"
        output_path = os.path.join("output", output_name)
        os.makedirs("output", exist_ok=True)
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        return output_path

class PoseAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pose Analyzer")
        self.pose_analyzer = PoseAnalyzer(mode='3d')  # Default to 3D mode
        self.process_joints = {}
        self.analyze_joints = {}
        self.create_widgets()

    def create_widgets(self):

        # Add a mode selection dropdown
        self.mode_var = tk.StringVar(value='3d')
        mode_label = tk.Label(self.master, text="Analysis Mode:")
        mode_label.pack()
        mode_dropdown = ttk.Combobox(self.master, textvariable=self.mode_var, values=['2d', '3d'])
        mode_dropdown.pack()
        mode_dropdown.bind('<<ComboboxSelected>>', self.update_mode)

        self.select_file_button = tk.Button(self.master, text="Select File (Image or Video)", command=self.select_file)
        self.select_file_button.pack()

        self.process_file_button = tk.Button(self.master, text="Process File", command=self.process_file)
        self.process_file_button.pack()

        self.analyze_data_button = tk.Button(self.master, text="Analyze Data", command=self.analyze_data)
        self.analyze_data_button.pack()

        self.record_webcam_button = tk.Button(self.master, text="Record from Webcam", command=self.record_webcam)
        self.record_webcam_button.pack()

        self.status_label = tk.Label(self.master, text="Ready")
        self.status_label.pack()

        self.progress_bar = ttk.Progressbar(self.master, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack()

        # Create two separate sections for processing and analysis joints
        processing_frame = ttk.LabelFrame(self.master, text="Joints to Process")
        processing_frame.pack(padx=10, pady=5, fill="x")

        analysis_frame = ttk.LabelFrame(self.master, text="Joints to Analyze")
        analysis_frame.pack(padx=10, pady=5, fill="x")

        for angle in self.pose_analyzer.config['angles_to_display']:
            self.process_joints[angle] = tk.BooleanVar(value=True)
            ttk.Checkbutton(processing_frame, text=angle, variable=self.process_joints[angle]).pack(anchor="w")

            self.analyze_joints[angle] = tk.BooleanVar(value=True)
            ttk.Checkbutton(analysis_frame, text=angle, variable=self.analyze_joints[angle]).pack(anchor="w")

        ttk.Label(self.master, text="Note: Analysis will only include joints that were detected during processing.").pack(pady=5)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            initialdir="./input", 
            title="Select file",
            filetypes=(("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if self.file_path:
            messagebox.showinfo("File Selected", f"Selected file: {self.file_path}")

    def process_file(self):
        if hasattr(self, 'file_path'):
            self.pose_analyzer.file_path = self.file_path
            output_path = self.pose_analyzer.get_output_path(self.file_path)
            
            joints_to_process = [angle for angle, var in self.process_joints.items() if var.get()]
            
            if self.file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.progress_bar["value"] = 0
                self.master.update()
                
                for progress in self.pose_analyzer.process_video(self.file_path, output_path, joints_to_process):
                    self.progress_bar["value"] = progress
                    self.master.update()
                
                success = True
                message = f"Video processing complete. Saved as {output_path}"
            else:
                success, message = self.pose_analyzer.process_file(self.file_path, output_path, joints_to_process)
            
            if success:
                messagebox.showinfo("Processing Complete", message)
                self.analyze_data_button.config(state='normal')
            else:
                messagebox.showerror("Processing Error", message)
        else:
            messagebox.showerror("Error", "Please select a file first")
            
    def update_mode(self, event=None):
        new_mode = self.mode_var.get()
        if new_mode != self.pose_analyzer.mode:
            self.pose_analyzer.mode = new_mode
            print(f"Analysis mode updated to: {self.pose_analyzer.mode}")

            # Reset data and variables
            self.pose_analyzer.angles_data = []
            self.pose_analyzer.detected_joints.clear()
            self.pose_analyzer.file_path = None

            # Update UI elements
            self.status_label.config(text=f"Mode changed to {new_mode.upper()}. Please select a new file.")
            self.process_file_button.config(state='normal')  # Enable the Process File button
            self.analyze_data_button.config(state='disabled')

            # Update MediaPipe configuration
            self.pose_analyzer.pose = self.pose_analyzer.configure_mediapipe()

            # Update joint checkboxes based on the new mode
            self.update_joint_checkboxes()

            # Clear any existing results or visualizations
            self.clear_results()

            messagebox.showinfo("Mode Changed", f"Analysis mode has been changed to {new_mode.upper()}. "
                                                f"Please select a new file to process.")
            
    def update_joint_checkboxes(self):
        # This method would update the joint checkboxes based on the current mode
        # For example, you might want to disable certain joints that are not relevant in 2D mode
        for angle in self.process_joints:
            if self.pose_analyzer.mode == '2d' and angle.endswith('_Z'):  # Assuming Z-axis related joints
                self.process_joints[angle].set(False)
                self.analyze_joints[angle].set(False)
            else:
                self.process_joints[angle].set(True)
                self.analyze_joints[angle].set(True)

    def clear_results(self):
        # Clear any existing results, plots, or output files
        if hasattr(self, 'result_window'):
            self.result_window.destroy()
        
        # Clear output files (you might want to ask for confirmation before deleting)
        output_dir = "angle_data"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        print("Previous results cleared.")

    def analyze_data(self):
        selected_angles = [angle for angle, var in self.analyze_joints.items() if var.get()]
        success, message = self.pose_analyzer.analyze_data(selected_angles)
        if success:
            messagebox.showinfo("Analysis Complete", message)
        else:
            messagebox.showerror("Analysis Error", message)

    def record_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access webcam")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        recording = False
        output_filename = None

        try:
            with self.pose_analyzer.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:
                        self.pose_analyzer.mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, self.pose_analyzer.mp_pose.POSE_CONNECTIONS)

                    if recording:
                        out.write(image)

                    status_text = "Recording" if recording else "Press SPACE to start recording"
                    cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow('Webcam (SPACE: start/stop, ESC: quit)', image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break
                    elif key == 32:  # SPACE key
                        if not recording:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            os.makedirs("input", exist_ok=True)
                            output_filename = os.path.join("input", f"webcam_recording_{timestamp}.mp4")
                            out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                            recording = True
                            print("Recording started...")
                        else:
                            recording = False
                            out.release()
                            out = None
                            print(f"Recording stopped. Saved as {output_filename}")
                            print(f"File exists: {os.path.exists(output_filename)}")
                            print(f"File size: {os.path.getsize(output_filename)} bytes")

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        if output_filename and os.path.exists(output_filename) and os.path.getsize(output_filename) > 1000:
            self.file_path = output_filename
            self.pose_analyzer.file_path = output_filename
            message = f"Video saved as: {output_filename}\n\nYou can now use the 'Process File' button to analyze this video."
            messagebox.showinfo("Recording Complete", message)
            self.status_label.config(text="Ready to process recorded video")
            self.process_file_button.config(state='normal')
        else:
            messagebox.showerror("Recording Error", "Failed to save the recorded video or file is too small.")

def main():
    root = tk.Tk()
    app = PoseAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
