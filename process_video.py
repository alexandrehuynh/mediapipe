import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import Tk, filedialog

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def put_text_with_background(img, text, position, font, font_scale, text_color, thickness):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    bg_rect_x1 = position[0]
    bg_rect_y1 = position[1] - text_size[1] - 10
    bg_rect_x2 = position[0] + text_size[0] + 10
    bg_rect_y2 = position[1]
    
    cv2.rectangle(img, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), (0, 0, 0), -1)
    
    cv2.putText(img, text, (position[0] + 5, position[1] - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

# Function to select video file
def select_video():
    print("Opening file dialog...")
    root = Tk()
    root.withdraw()  # Hide the main window
    video_path = filedialog.askopenfilename(initialdir="./input", 
                                            title="Select video file",
                                            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    print(f"Selected path: {video_path}")
    return video_path

# Function to get output path
def get_output_path(input_path):
    base_name = os.path.basename(input_path)
    output_name = f"mapped_{base_name}"
    output_path = os.path.join("output", output_name)
    return output_path

# Select video file
video_path = select_video()
if not video_path:
    print("No video selected. Exiting...")
    exit()

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
output_path = get_output_path(video_path)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # New points for additional angles
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            
            # Calculate angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle(left_ear, left_shoulder, left_elbow)
            left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_ankle_angle = calculate_angle(left_knee, left_ankle, left_heel)
            
            # Visualize angles with larger, more visible text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_color = (255, 255, 255)  # White color

            shoulder_position = tuple(np.multiply(left_shoulder, [frame_width, frame_height]).astype(int))
            elbow_position = tuple(np.multiply(left_elbow, [frame_width, frame_height]).astype(int))
            wrist_position = tuple(np.multiply(left_wrist, [frame_width, frame_height]).astype(int))
            hip_position = tuple(np.multiply(left_hip, [frame_width, frame_height]).astype(int))
            knee_position = tuple(np.multiply(left_knee, [frame_width, frame_height]).astype(int))
            ankle_position = tuple(np.multiply(left_ankle, [frame_width, frame_height]).astype(int))

            put_text_with_background(image, f"Shoulder: {left_shoulder_angle:.0f}", shoulder_position, font, font_scale, text_color, thickness)
            put_text_with_background(image, f"Elbow: {left_elbow_angle:.0f}", elbow_position, font, font_scale, text_color, thickness)
            put_text_with_background(image, f"Wrist: {left_wrist_angle:.0f}", wrist_position, font, font_scale, text_color, thickness)
            put_text_with_background(image, f"Hip: {left_hip_angle:.0f}", hip_position, font, font_scale, text_color, thickness)
            put_text_with_background(image, f"Knee: {left_knee_angle:.0f}", knee_position, font, font_scale, text_color, thickness)
            put_text_with_background(image, f"Ankle: {left_ankle_angle:.0f}", ankle_position, font, font_scale, text_color, thickness)
            
        except:
            pass

        # Draw the pose annotation on the image
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Display the image
        cv2.imshow('MediaPipe Pose', image)

        # Save the frame
        out.write(image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to: {output_path}")