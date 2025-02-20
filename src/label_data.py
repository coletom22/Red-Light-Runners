import cv2
import os

# Standardized image width/height
IMAGE_WIDTH, IMAGE_HEIGHT = 768, 448

# Directory to pull raw images from
raw_dir = "../data/images/raw"
# Directory to place images once processed (resized and annotated)
processed_dir = "../data/images/processed"

# Color mapping based on key presses
color_mapping = {
    ord("1"): (0, 255, 0),   # Green
    ord("2"): (0, 0, 255),   # Red
    ord("3"): (0, 255, 255), # Yellow
}

# Label mapping based on colors
label_mapping = {
    (0, 255, 0): "Green Light",
    (0, 0, 255): "Red Light",
    (0, 255, 255): "Yellow Light"
}

# Class mapping based on label (for YOLO format that uses int instead of str)
class_mapping = {
    "Green Light": 1,
    "Red Light": 2,
    "Yellow Light": 3
}

# Mouse callback function
def draw_rectangle_with_drag(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, img, current_color

    # Left clicking starts a drawing event if the user is not currently drawing
    if event == cv2.EVENT_LBUTTONDOWN and not drawing:  
        drawing = True # event status is drawing
        ix, iy = x, y # anchor point for the first corner of the rectangle
        img_copy = img.copy()  # reset copy when starting a new rectangle

    # When the cursor is moving and we are in drawing status display adjusted size of rectangle based on cursor location
    elif event == cv2.EVENT_MOUSEMOVE and drawing:  
        img_copy = img.copy()  # reset to avoid multiple overlapping rectangles
        cv2.rectangle(img_copy, (ix, iy), (x, y), current_color, 1) # drawing rectangle from ix, iy to current cursor position

    # Left click when we are already drawing places the rectangle where the cursor is located during the click
    elif event == cv2.EVENT_LBUTTONDOWN and drawing:  
        drawing = False # reset event status to not drawing
        cv2.rectangle(img, (ix, iy), (x, y), current_color, 1)  # draw on final image
        annotations.append({
            "x1": min(ix, x),
            "x2": max(ix, x),
            "y1": min(iy, y),
            "y2": max(iy, y),
            "color_code": current_color,
            "color": label_mapping[current_color],
            "class": class_mapping[label_mapping[current_color]]
        }) # appends a map of values needed for documentation min/max x and y coordinates, color codes, colors, and class
        
        redraw_rectangles()  # for view consistency

# redraws all the rectangles from the list of annotations
def redraw_rectangles():
    global img, img_copy
    img = resized_img.copy()  # reset to the resized image
    for ann in annotations:
        cv2.rectangle(img, 
                      (ann["x1"], ann["y1"]), 
                      (ann["x2"], ann["y2"]), 
                      ann["color_code"], 1) # draws each rectangle
        
    img_copy = img.copy()  # Update the display copy

# Iterate over each file in the raw data dir
for filename in os.listdir(raw_dir):
    file_path = os.path.join(raw_dir, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        original_img = cv2.imread(file_path)  # keep original image
        if original_img is None:
            raise FileNotFoundError("Image not found. Check the file path.")
        
        # Resize the image to a uniform size
        resized_img = cv2.resize(original_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Initialize working images
        img = resized_img.copy()   # Active drawing image
        img_copy = img.copy()      # Image copy for real-time updates

        # Anchor variables
        ix, iy = -1, -1
        drawing = False
        current_color = (0, 255, 0)  # Default: Green

        # Prepare annotation file
        img_filename = os.path.basename(file_path)
        text_filename = os.path.splitext(img_filename)[0] + ".txt"
        annotation_path = f"../data/labels/viewing/{text_filename}"

        # List to store dicts of annotations
        annotations = []
        # Create window and set mouse callback
        cv2.namedWindow("Label Data", cv2.WINDOW_NORMAL)  # Allow resizing
        cv2.resizeWindow("Label Data", IMAGE_WIDTH, IMAGE_HEIGHT)  # Set initial size
        cv2.setMouseCallback("Label Data", draw_rectangle_with_drag)

        # Display loop
        while True:
            cv2.imshow("Label Data", img_copy)  # Show dynamic updates
            key = cv2.waitKey(10) & 0xFF
            
            # Press 'Esc' to exit
            if key == 27:
                break
            
            # Press 's' to save annotations
            elif key == ord("s"):

                # Open annotation text file, iterate over annotations and write to text file
                with open(annotation_path, 'w') as annotation_file:
                    for annotation in annotations:
                        annotation_file.write(f"{annotation}\n")
                print(f"Annotations saved to {annotation_path}")

                # Open processed_dir and write resized image to the directory
                os.makedirs(processed_dir, exist_ok=True)
                processed_path = os.path.join(processed_dir, os.path.basename(file_path))
                cv2.imwrite(processed_path, resized_img)
                print(f"Moved {file_path} -> {processed_path}")

                break

            # Change rectangle color based on number key
            elif key in color_mapping:  
                current_color = color_mapping[key]
                print(f"Class changed to: {label_mapping[color_mapping[key]]}")
            
            # Press 'z' to undo last rectangle
            elif key == ord("z") and annotations: 
                annotations.pop()  # remove last rectangle
                redraw_rectangles()  # reset image and redraw remaining rectangles
                print("Last rectangle removed!")

        cv2.destroyAllWindows()
