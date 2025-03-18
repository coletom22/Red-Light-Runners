import json
import os
import cv2

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

# resets selected annotations to none
def reset_selection(annotations):
    for ann in annotations:
        ann['thickness'] = 1


# removes selected/'highlighted' (bbox with thickness 2)
def remove_bbox(annotations, model_name, img_name):
    for ann in annotations[:]:
        if ann['thickness'] == 2:
            update_annotation_data(model_name, img_name, 'rmv_' + ann['color'], 1)
            annotations.remove(ann)
    print(annotations)


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
