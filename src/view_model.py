import os
from ultralytics import YOLO
import cv2

class_color = {
    'red_light': (0,0,255),
    'green_light': (0,255,0),
    'yellow_light': (0,255,255)
}

video_path_out = r'output_2.mp4'
# Open video capture (replace with the path to your video file)
video_path = '../data/videos/processed/20250222_152040M.mp4'


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

ret, frame = cap.read()
if frame is None:
    print("Error: Unable to read the first frame from the video.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model = YOLO("../models/first_test_202502281223.pt")
threshold = 0.0

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), class_color[results.names[int(class_id)]], 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, class_color[results.names[int(class_id)]], 3, cv2.LINE_AA)
        
    out.write(frame)
    ret, frame = cap.read()

# Wait for a key press before closing OpenCV windows

cap.release()
out.release()
