import os
import shutil

from sklearn.model_selection import train_test_split


# function to move data out of model training stage
def dump_data(img_dest='../data/images/processed', label_dest='../data/labels/formatted'):
    img_dest_dir = img_dest
    label_dest_dir = label_dest

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