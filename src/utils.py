import cv2
import os
import shutil
from ultralytics import YOLO
import sys
from sklearn.model_selection import train_test_split


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

    label_dir = '../data/model_data/labels'

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

    if os.path.exists(os.path.join(label_dir, 'train.cache')):
        os.remove(os.path.join(label_dir, 'train.cache'))
    if os.path.exists(os.path.join(label_dir, 'validation.cache')):
        os.remove(os.path.join(label_dir, 'validation.cache'))


def reset_selection(annotations):
    for ann in annotations:
        ann['thickness'] = 1
    print("Annotations reset")

def remove_bbox(annotations):
    for ann in annotations[:]:
        if ann['thickness'] == 2:
            annotations.remove(ann)
    print(annotations)

# function that moves files to train and validation dirs
def train_val_split(img_source_dir, label_source_dir, img_train_dir, label_train_dir,  img_val_dir, label_val_dir, train_percent=0.8):
    img_files = os.listdir(img_source_dir)

    train_files, val_files = train_test_split(img_files, test_size = 1-train_percent, random_state=42)

    for file in train_files:
        img_source_path = os.path.join(img_source_dir, file)
        img_dest_path = os.path.join(img_train_dir, file)

        label_file = os.path.splitext(file)[0] + '.txt'
        label_source_path = os.path.join(label_source_dir, label_file)
        label_dest_path = os.path.join(label_train_dir, label_file)
        try:
            shutil.move(img_source_path, img_dest_path)
            print(f"Moved {file} to image/train/")
            shutil.move(label_source_path, label_dest_path)
            print(f"Moved {label_file} to label/train/")

        except Exception as e:
            print(f"Error moving {file} to image/train/ and {label_file} to label/train/: {e}")

    for file in val_files:
        img_source_path = os.path.join(img_source_dir, file)
        img_dest_path = os.path.join(img_val_dir, file)

        label_file = os.path.splitext(file)[0] + '.txt'
        label_source_path = os.path.join(label_source_dir, label_file)
        label_dest_path = os.path.join(label_val_dir, label_file)
        try:
            shutil.move(img_source_path, img_dest_path)
            print(f"Moved {file} to image/val/")
            shutil.move(label_source_path, label_dest_path)
            print(f"Moved {label_file} to label/val/")

        except Exception as e:
            print(f"Error moving {file} to image/val/ and {label_file} to label/val/: {e}")
        

def train_assistant_model(model_name):
    img_source_dir = '../data/images/pretrain'
    label_source_dir = '../data/labels/formatted'

    img_train_dir = '../data/model_data/images/train'
    img_val_dir = '../data/model_data/images/validation'

    label_train_dir = '../data/model_data/labels/train'
    label_val_dir = '../data/model_data/labels/validation'

    weight_dir = "../models/current_assistant"
    train_val_split(img_source_dir, label_source_dir, img_train_dir, label_train_dir, img_val_dir, label_val_dir)

    model = YOLO("yolov8n.yaml")
    results = model.train(
        data = "../notebooks/data.yaml", 
        epochs=30, 
        imgsz=768, 
        device=0, 
        rect=True, 
        project="../runs", 
        name=f"{model_name}"
    )

    for weight in os.listdir(weight_dir):
        shutil.move(os.path.join(weight_dir, weight), os.path.join("../models", weight))

    weight_src_path = os.path.join("../runs", f"{model_name}", "weights", "best.pt")
    weight_dst_path = os.path.join(weight_dir, f"{model_name}.pt")

    shutil.move(weight_src_path, weight_dst_path)
    
