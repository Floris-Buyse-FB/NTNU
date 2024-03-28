from packages import log, check_directory, get_image_paths, load_model, remove_old_directories, remove_old_runs, predict, width_length_difference
from ultralytics import settings
import numpy as np
import shutil
import torch
import cv2
import os

# CONSTANTS
IMG_PATH = './images/classification/random'
SAVE_DIR = './images/cropped_scales/random'
FILENAME = os.path.basename(__file__)

MODEL_PATH= './models/crop_random_scale.pt'

RUNS_DIR = settings['runs_dir']
PREDICT_PATH = os.path.join(RUNS_DIR, 'obb/predict')


def process_results(results):
    for result_list in results:
        for result in result_list:
            image_path = result.path
            image_name = os.path.basename(image_path)

            if len(result.obb.xyxyxyxy) > 1:
                differences = torch.tensor([width_length_difference(box) for box in result.obb.xyxyxyxy])
                box_with_largest_difference_index = differences.argmax().item()
                bbox = result.obb.xyxyxyxy[box_with_largest_difference_index]
            else:
                bbox = result.obb.xyxyxyxy[0]
            
            image = cv2.imread(image_path)
            quadrilateral = np.array(bbox.cpu() if torch.cuda.is_available() else bbox)
            x, y, w, h = cv2.boundingRect(quadrilateral)
            cropped_image = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(SAVE_DIR, image_name.replace('.jpg', '_scale_only.jpg')), cropped_image)
            shutil.move(image_path, os.path.join(SAVE_DIR, image_name))


def main():
    try:
        check_directory(SAVE_DIR)
    except Exception as e:
        log(f'Error creating directory: {e}', FILENAME)

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        log(f'Error loading model: {e}', FILENAME)
    
    try:
        images = get_image_paths(IMG_PATH)
    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)

    try:
        remove_old_runs('obb')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)

    try:
        results = predict(model, images, conf=0.7, verbose=False)
    except Exception as e:
        log(f'Error predicting images: {e}', FILENAME)

    try:
        process_results(results)
    except Exception as e:
        log(f'Error processing results: {e}', FILENAME)
        
    try:
        remove_old_directories([IMG_PATH, PREDICT_PATH])
    except Exception as e:
        log(f'Error removing directories: {e}', FILENAME)


if __name__ == '__main__':
    main()