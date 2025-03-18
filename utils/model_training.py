import os
import shutil
import cv2
from ultralytics import YOLO
import sys
sys.path.append(os.path.abspath('../utils'))
import data_management

# automatically called when enough new labeled images are in processed
# trains a new model and inserts it as the new assistant for labeling
def train_assistant_model(model_name):
    img_source_dir = '../data/images/processed'
    label_source_dir = '../data/labels/formatted'

    img_train_dir = '../data/model_data/images/train'
    img_val_dir = '../data/model_data/images/validation'

    label_train_dir = '../data/model_data/labels/train'
    label_val_dir = '../data/model_data/labels/validation'

    weight_dir = "../models/current_assistant"
    data_management.train_val_split(img_source_dir, label_source_dir, img_train_dir, label_train_dir, img_val_dir, label_val_dir)

    print(f"Training set size: {len(img_train_dir)} | Validation set size: {len(img_val_dir)}\n")
    model = YOLO("yolov8n.yaml")
    results = model.train(
        data = "../SLD.yaml", 
        epochs=30, 
        imgsz=768, 
        device=0, 
        project="../runs", 
        name=f"{model_name}"
    )

    # any files that are in the current_assistant dir need to be moved to models/
    for weight in os.listdir(weight_dir):
        shutil.move(os.path.join(weight_dir, weight), os.path.join("../models", weight))

    # the newly trained model is going to be set as the current_assistant
    weight_src_path = os.path.join("../runs", f"{model_name}", "weights", "best.pt")
    weight_dst_path = os.path.join(weight_dir, f"{model_name}.pt")
    
    shutil.move(weight_src_path, weight_dst_path)


def predict_video(vid_path, output_path, model_path, thresh=0.0):
    class_color = {
        'red_light': (0,0,255),
        'green_light': (0,255,0),
        'yellow_light': (0,255,255)
    }
    video_path = vid_path
    video_path_out = output_path

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
    model = YOLO(model_path)
    threshold = thresh

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
    cap.release()
    out.release()