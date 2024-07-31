import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import Tk, filedialog

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# Select video file
video_path = select_video()
if not video_path:
    print("No video selected. Exiting...")
    exit()

# Function to get output path
def get_output_path(input_path):
    base_name = os.path.basename(input_path)
    output_name = f"mapped_{base_name}"
    output_path = os.path.join("output", output_name)
    return output_path


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