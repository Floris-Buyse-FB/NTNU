import sys
sys.path.append('/home/floris/Projects/NTNU/packages')

from ultralytics import YOLO, settings
import os
import shutil
from packages import log
import torch


# CONSTANTS
IMG_PATH_FIXED = './images/scale_fixed'
IMG_PATH_RANDOM = './images/scale_random'
MODEL_PATH_OBB = './models/classify_scale_nano.pt'
CLASSIFICATION_DIR = './images/classification'

def create_classification_dirs():
    try:
        os.makedirs(CLASSIFICATION_DIR, exist_ok=True)
        os.makedirs(os.path.join(CLASSIFICATION_DIR, 'fixed'), exist_ok=True)
        os.makedirs(os.path.join(CLASSIFICATION_DIR, 'random'), exist_ok=True)
        os.makedirs(os.path.join(CLASSIFICATION_DIR, 'no_class'), exist_ok=True)
    except OSError as e:
        log(f'Error: {e}')
        print('Something went wrong, check the log file for more information')

def load_model():
    try:
        return YOLO(MODEL_PATH_OBB)
    except Exception as e:
        log(f'Error loading models: {e}')
        print('Something went wrong, check the log file for more information')

def get_image_paths(img_path):
    try:
        return [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))]
    except Exception as e:
        log(f'Error reading images: {e}')
        print('Something went wrong, check the log file for more information')

def remove_old_runs():
    try:
        shutil.rmtree(os.path.join(settings['runs_dir'], 'obb'), ignore_errors=True)
    except Exception as e:
        log(f'Error removing old runs: {e}')
        print('Something went wrong, check the log file for more information')

def predict_images(model, images):
    try:
        return model(images, conf=0.8, verbose=False)  # conf=0.8: only images with a confidence of 80% or more
    except Exception as e:
        log(f'Error predicting images: {e}')
        print('Something went wrong, check the log file for more information')

def move_images(results):
    for result in results:
        try:
            class_label = result.obb.cls  # tensor with 1 item if detected, no item if not detected
            path = result.path
            if len(class_label) > 0:
                if class_label[0] == 0:
                    shutil.move(path, os.path.join(CLASSIFICATION_DIR, 'fixed', os.path.basename(path)))
                elif class_label[0] == 1:
                    shutil.move(path, os.path.join(CLASSIFICATION_DIR, 'random', os.path.basename(path)))
            else:
                shutil.move(path, os.path.join(CLASSIFICATION_DIR, 'no_class', os.path.basename(path)))
        except Exception as e:
            log(f'Error moving images: {e}')
            print('Something went wrong, check the log file for more information')

def remove_old_directories():
    try:
        shutil.rmtree(IMG_PATH_FIXED)
        shutil.rmtree(IMG_PATH_RANDOM)
    except Exception as e:
        log(f'Error removing old directories: {e}')
        print('Something went wrong, check the log file for more information')

def main():
    create_classification_dirs()
    model_obb = load_model()
    all_images_fixed = get_image_paths(IMG_PATH_FIXED)
    all_images_random = get_image_paths(IMG_PATH_RANDOM)
    combined_images = all_images_fixed + all_images_random
    remove_old_runs()
    if len(combined_images) > 50:
        for i in range(0, len(combined_images), 50):
            results = predict_images(model_obb, combined_images[i:i+50])
            move_images(results)
            torch.cuda.empty_cache()
    else:
        results = predict_images(model_obb, combined_images)
        move_images(results)
    remove_old_directories()

if __name__ == '__main__':
    main()