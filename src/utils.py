import cv2
import os
import shutil
# variables
# Standardized image width/height
IMAGE_WIDTH, IMAGE_HEIGHT = 768, 448

# function for converting rectangle points and class value to YOLO format
def viewing_to_yolo(annotations, IMAGE_WIDTH, IMAGE_HEIGHT) -> list[dict]:
    yolo_annotations = []

    for ann in annotations:
        x1, x2, y1, y2, class_id = ann['x1'], ann['x2'], ann['y1'], ann['y2'], ann['class']

        # compute yolo format values
        x_center = (x1 + x2) / 2 / IMAGE_WIDTH
        y_center = (y1 + y2) / 2 / IMAGE_HEIGHT
        width = (x2 - x1) / IMAGE_WIDTH
        height = (y2 - y1) / IMAGE_HEIGHT

        yolo_annotations.append((class_id, round(x_center, 6), round(y_center,6), round(width, 6), round(height, 6)))

    return yolo_annotations

# function for converting a video into individual frames and storing in separate directory
def video_to_frames(video_path, target_directory, video_name):
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
        
        frame_filename = os.path.join(target_directory, f"{video_name}_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames at {fps} FPS.")

# function to move data out of model training stage
def dump_data():
    img_dest_dir = '../data/images/processed'
    label_dest_dir = '../data/labels/formatted'

    img_train_dir = '../data/model_data/images/train'
    img_val_dir = '../data/model_data/images/validation'

    label_train_dir = '../data/model_data/labels/train'
    label_val_dir = '../data/model_data/labels/validation'

    for file in os.listdir(img_train_dir):
        try:
            shutil.move(os.path.join(img_train_dir, file), os.path.join(img_dest_dir, file))

        except Exception as e:
            print(f"Could not move {file}")

    for file in os.listdir(img_val_dir):
        try:
            shutil.move(os.path.join(img_val_dir, file), os.path.join(img_dest_dir, file))

        except Exception as e:
            print(f"Could not move {file}")
    
    for file in os.listdir(label_train_dir):
        try:
            shutil.move(os.path.join(label_train_dir, file), os.path.join(label_dest_dir, file))

        except Exception as e:
            print(f"Could not move {file}")
        
    for file in os.listdir(label_val_dir):
        try:
            shutil.move(os.path.join(label_val_dir, file), os.path.join(label_dest_dir, file))

        except Exception as e:
            print(f"Could not move {file}")