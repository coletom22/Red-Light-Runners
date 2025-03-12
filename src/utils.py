import cv2
import os
import shutil
import json

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

# function similar to video_to_frames except used for extracting specific frames for testing
def create_test_frames(video_path, target_directory, video_name, window_width, window_height):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Video does not exist")
        return
    
    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Unknown FPS")
        return
    
    # Ensure target directory exists
    os.makedirs(target_directory, exist_ok=True)
    window_name = f"Press 's' to save frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
    cv2.resizeWindow(window_name, window_width, window_height)  # Set to 95% of screen size
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imshow(window_name, resized_frame)
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press
        
        if key == ord('s'):  # Save the frame if 's' is pressed
            frame_filename = os.path.join(target_directory, f"{video_name}_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            print(f"Saved: {frame_filename}")
            frame_count += 1
        
        elif key == ord('q'):  # Quit the process if 'q' is pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Process completed. Saved {frame_count} frames.")


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

# resets selected annotations to none
def reset_selection(annotations):
    for ann in annotations:
        ann['thickness'] = 1

# keeps track of various annotation metrics ('key') organized by model
def update_annotation_data(model_name, img_name, key, increment=1):
    file_path = f'../annotation_meta_data/{model_name}.json'

    if not os.path.exists(file_path):
        data = {}
    else:
        try:
            with open(f'../annotation_meta_data/{model_name}.json', 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error with file {model_name}.json")
            data = {}

    if img_name not in data:
        data[img_name] = {"total_annotations": 0}
    
    if key not in data[img_name]:
        data[img_name][key] = 0
    
    data[img_name][key] += increment

    with open(f'../annotation_meta_data/{model_name}.json', 'w') as file:
        json.dump(data, file, indent = 4)

    print(f"Updated {model_name}.json data successfully")

# removes selected/'highlighted' (bbox with thickness 2)
def remove_bbox(annotations, model_name, img_name):
    for ann in annotations[:]:
        if ann['thickness'] == 2:
            update_annotation_data(model_name, img_name, 'rmv_' + ann['color'], 1)
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
            shutil.move(label_source_path, label_dest_path)

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
            shutil.move(label_source_path, label_dest_path)

        except Exception as e:
            print(f"Error moving {file} to image/val/ and {label_file} to label/val/: {e}")
        
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
    train_val_split(img_source_dir, label_source_dir, img_train_dir, label_train_dir, img_val_dir, label_val_dir)

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
    

# writes all the images being used in the training to a text file
def track_dataset(model_name, training_dir):
    training_dataset = os.listdir(training_dir)
    with open("../models/meta_data/training_data_info.txt", 'a') as file:
        file.write(f"{model_name}\n")
        file.write(f"Size: {len(training_dataset)}\n")
        file.write("\n".join(training_dataset))
        file.write('\n')
    print(f"Model: {model_name} dataset saved to '../models/meta_data/training_data_info.txt'\n")


# Retrieves the latest training's dataset size
def get_latest_dataset_size():
    try:
        with open("../models/meta_data/training_data_info.txt", 'r') as file:
            lines = file.readlines()

        if lines:
            size_index = -1

            for i in range(len(lines)-1, -1, -1):
                if lines[i].startswith("Size:"):
                    size_index = i
                    break
            if size_index != -1:
                dataset_size = int(lines[size_index].strip().split(": ")[1])
                return dataset_size
        else:
            return None
    except FileNotFoundError:
        print("The file for training data does not exist")
        return None
    

# Opens the meta data text file and retrieves the most recently trained models meta data
def get_latest_dataset_info():
    try:
        with open("../models/meta_data/training_data_info.txt", 'r') as file:
            lines = file.readlines()

        if lines:
            size_index = -1

            for i in range(len(lines)-1, -1, -1):
                if lines[i].startswith("Size:"):
                    size_index = i
                    break
            if size_index != -1:
                model_name = lines[size_index-1].strip()
                dataset_size = int(lines[size_index].strip().split(": ")[1])
                dataset_files = lines[size_index+1 : size_index + 1 + dataset_size]
                dataset_files = [file.strip() for file in dataset_files]
                return model_name, dataset_size, dataset_files
        else:
            return None, None, None
    except FileNotFoundError:
        print("The file for training data does not exist")
        return None, None, None
    
# CREATE STRATIFY FUNC THAT ENABLES MORE DIVERSE TRAINING
def stratify():
    print("strat")


# function for viewing annotations based on YOLO format
def view_yolo_ann(img_dir, label_dir):
    import screeninfo

    # Get second monitor details
    monitors = screeninfo.get_monitors()
    if len(monitors) > 1:
        second_monitor = monitors[1]  # Assuming second monitor is at index 1
        x_offset, y_offset = second_monitor.x, second_monitor.y
        screen_width, screen_height = second_monitor.width, second_monitor.height
    else:
        x_offset, y_offset = 0, 0  # Default to primary monitor
        screen_width, screen_height = monitors[0].width, monitors[0].height  # Primary monitor size

    # Reduce size slightly to keep windowed view
    window_width = int(screen_width * 0.95)  # 95% of the screen width
    window_height = int(screen_height * 0.95)  # 95% of the screen height

    color_map = {
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (0, 255, 255)
    }
    files = os.listdir(img_dir)
    for filename in files:
        img = cv2.imread(os.path.join(img_dir, filename))

        height, width, _ = img.shape
        label_path = os.path.join(label_dir, filename.split(".")[0] + '.txt')

        with open(label_path, 'r') as label_file:
            annotations = []
            for line in label_file:
                ann_map = {}
                keys = line.split(' ')
                ann_map['class'] = int(keys[0])
                ann_map['x_center'] = float(keys[1])
                ann_map['y_center'] = float(keys[2])
                ann_map['width'] = float(keys[3])
                ann_map['height'] = float(keys[4].replace('\n', ''))
                annotations.append(ann_map)
        print(annotations)
        for ann in annotations:
            x1 = ann['x_center'] - (ann['width']/2)
            y1 = ann['y_center']- (ann['height']/2)
            x2 = ann['x_center'] + (ann['width']/2)
            y2 = ann['y_center'] + (ann['height']/2)
            cv2.rectangle(img, (int(width*x1), int(height*y1)), (int(width*x2), int(height*y2)), color_map[ann['class']])

        window_name = f"{filename} with annotations"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
        cv2.resizeWindow(window_name, window_width, window_height)  # Set to 95% of screen size
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def predict_video(vid_path, output_path):
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
    model_name = os.listdir("../models/current_assistant")[0]
    print(model_name)
    model = YOLO(f"../models/current_assistant/{model_name}")
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
    cap.release()
    out.release()


def iou(pred, gt):
    x1, y1, w1, h1, score = pred
    x2, y2, w2, h2 = gt

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)

    intersection_area = max(0, xi2-xi1) * max(0, yi2-yi1)

    pred_area = w1 * h1
    gt_area = w2 * h2
    union_area = pred_area + gt_area - intersection_area
    
    return intersection_area / union_area if union_area else 0


