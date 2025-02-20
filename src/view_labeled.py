import ast
import cv2
import os

# Standardized image width/height
IMAGE_WIDTH, IMAGE_HEIGHT = 768, 448

# Directory to pull images from
image_dir = "../data/images/processed"
# Directory to pull labels from
label_dir = "../data/labels/viewing"

# For each file in the images directory
for filename in os.listdir(image_dir):
    # Create full path for image and label
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

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
    cv2.imshow(f"{filename} with annotations", img)
    # Press any key to move to next image
    cv2.waitKey(0)
    cv2.destroyAllWindows()