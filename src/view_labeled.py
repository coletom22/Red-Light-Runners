import ast
import cv2
import os

# Standardized image width/height
IMAGE_WIDTH, IMAGE_HEIGHT = 768, 448

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

# Center the window with some margin
window_x = x_offset + (screen_width - window_width) // 2
window_y = y_offset + (screen_height - window_height) // 2

# Directory to pull images from
image_dir = "../data/images/processed"
# Directory to pull labels from
label_dir = "../data/labels/viewing"

# For each file in the images directory
for filename in os.listdir(image_dir):
    # Create full path for image and label
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir,'viewing_' + os.path.splitext(filename)[0] + '.txt')

    # List that will store annotations from annotation text file
    annotations = []
    with open(label_path, 'r') as file:
        for line in file:
            annotations.append(ast.literal_eval(line.strip()))

    # load image
    img = cv2.imread(image_path)

    # Draw rectangles and write text based on annotations
    for annotation in annotations:
        cv2.rectangle(img, (annotation['x1'], annotation['y1']), (annotation['x2'], annotation['y2']), annotation['color_code'], 1)
        cv2.putText(img, annotation['color'] + ' - ' + str(annotation['class']), (max(10, annotation['x1']-10), max(20, annotation['y1']-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, annotation['color_code'], 1, cv2.LINE_AA)
    
    # Display image with annotations
    window_name = f"{filename} with annotations"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
    cv2.resizeWindow(window_name, window_width, window_height)  # Set to 95% of screen size
    cv2.moveWindow(window_name, window_x, window_y)  # Center it on the second monitor
    cv2.imshow(window_name, img)
    # Press any key to move to next image
    cv2.waitKey(0)
    cv2.destroyAllWindows()