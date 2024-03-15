from ultralytics import YOLO, settings
import os
import shutil
from packages import log

# CONSTANTS
IMG_PATH_FIXED = './images/scale_fixed'
IMG_PATH_RANDOM = './images/scale_random'
MODEL_PATH_OBB = './models/classify_scale_nano.pt'
CLASSIFICATION_DIR = './images/classification'

# CREATE CLASSIFICATION DIR
try:
    if not os.path.exists(CLASSIFICATION_DIR):
        os.makedirs(CLASSIFICATION_DIR)
    if not os.path.exists(os.path.join(CLASSIFICATION_DIR, 'fixed')):
        os.makedirs(os.path.join(CLASSIFICATION_DIR, 'fixed'))
    if not os.path.exists(os.path.join(CLASSIFICATION_DIR, 'random')):
        os.makedirs(os.path.join(CLASSIFICATION_DIR, 'random'))
    if not os.path.exists(os.path.join(CLASSIFICATION_DIR, 'no_class')):
        os.makedirs(os.path.join(CLASSIFICATION_DIR, 'no_class'))
except OSError as e:
    log(f'Error: {e}')
    print('Something went wrong, check the log file for more information')

# LOAD MODELS
try:
    model_obb = YOLO(MODEL_PATH_OBB)
except Exception as e:
    log(f'Error loading models: {e}')
    print('Something went wrong, check the log file for more information')

# GET ALL IMAGES
try:
    all_images_fixed = sorted(os.listdir(IMG_PATH_FIXED))
    all_images_random = sorted(os.listdir(IMG_PATH_RANDOM))
except Exception as e:
    log(f'Error reading images: {e}')
    print('Something went wrong, check the log file for more information')

# GET ALL IMAGES FULL PATHS
try:
    all_images_fixed = [os.path.join(IMG_PATH_FIXED, img) for img in all_images_fixed]
    all_images_random = [os.path.join(IMG_PATH_RANDOM, img) for img in all_images_random]

    combined_images = all_images_fixed + all_images_random
except Exception as e:
    log(f'Error reading images: {e}')
    print('Something went wrong, check the log file for more information')

# REMOVE OLD RUNS
try:
    shutil.rmtree(os.path.join(settings['runs_dir'], '\obb'), ignore_errors=True)
except Exception as e:
    log(f'Error removing old runs: {e}')
    print('Something went wrong, check the log file for more information')

# PREDICT IMAGES
try:
    results = model_obb(combined_images, conf=0.8) # conf=0.8: only images with a confidence of 80% or more
except Exception as e:
    log(f'Error predicting images: {e}')
    print('Something went wrong, check the log file for more information')

# MOVE IMAGES TO CLASSIFICATION DIR
for result in results:
    try:
        class_label = result.obb.cls # tensor with 1 item if detected, no item if not detected
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

# REMOVE OLD DIRECTORIES
try:
    shutil.rmtree(IMG_PATH_FIXED)
    shutil.rmtree(IMG_PATH_RANDOM)
except Exception as e:
    log(f'Error removing old directories: {e}')
    print('Something went wrong, check the log file for more information')