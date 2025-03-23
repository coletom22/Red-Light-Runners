import os
import cv2
import shutil
from dotenv import load_dotenv

from ultralytics import YOLO

load_dotenv()

IMAGE_WIDTH, IMAGE_HEIGHT = 768, 448

# function for converting a video into individual frames and storing in separate directory
def video_to_frames(video_path, target_directory, video_name, output_path):
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
    shutil.move(video_path, output_path)
    print(f"Moved {video_name} to processed")


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
                cv2.putText(frame, str(100 * round(score, 3)), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, class_color[results.names[int(class_id)]], 2, cv2.LINE_AA)
            
        out.write(frame)
        ret, frame = cap.read()
    cap.release()
    out.release()