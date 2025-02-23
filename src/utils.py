import cv2
import os
# function for converting rectangle points and class value to YOLO format
def viewing_to_yolo(annotations, IMAGE_WIDTH, IMAGE_HEIGHT) -> list[dict]:
    yolo_annotations = []

    for ann in annotations:
        x1, x2, y1, y2, class_id = ann['x1'], ann['x2'], ann['y1'], ann['y2'], ann['class_id']

        # compute yolo format values
        x_center = (x1 + x2) / 2 / IMAGE_WIDTH
        y_center = (y1 + y2) / 2 / IMAGE_HEIGHT
        width = (x2 - x1) / IMAGE_WIDTH
        height = (y2 - y1) / IMAGE_HEIGHT

        yolo_annotations.append((class_id, round(x_center, 6), round(y_center,6), round(width, 6), round(height, 6)))

    return yolo_annotations

# function for converting a video into individual frames and storing in separate directory
def video_to_frames(video_path, target_directory):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("video does not exist")
        return
    
    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("unknown FPS")
        return
    
    # Ensure target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(target_directory, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames at {fps} FPS.")
