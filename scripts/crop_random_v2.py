from ultralytics import YOLO
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from packages import log, width_length_difference, check_directory, get_image_paths, load_model, remove_old_runs, predict
import torch
import argparse

# CONSTANTS
SAVE_DIR= './images/cropped_scales/random'
FILENAME = os.path.basename(__file__)

MODEL_PATH = './models/crop_random_scale.pt'


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

            # Ensure the bounding box is within the image bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(x + w, image.shape[1])
            y2 = min(y + h, image.shape[0])

            cropped_image = image[y:y2, x:x2]
            cv2.imwrite(os.path.join(SAVE_DIR, image_name.replace('.jpg', '_scale_only.jpg')), cropped_image)
            shutil.move(image_path, os.path.join(SAVE_DIR, image_name))


def main(conf):
    try:
        check_directory(SAVE_DIR)
    except Exception as e:
        log(f'Error creating directory: {e}', FILENAME)

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        log(f'Error loading model: {e}', FILENAME)
    
    try:
        images = get_image_paths(SAVE_DIR, ext='.jpg', neg_ext='_scale_only.jpg')
        images_check = [img for img in images if not os.path.exists(img.replace('.jpg', '_scale_only.jpg'))]

        if len(images_check) == 0:
            print('No images to process')
            return

    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)

    try:
        remove_old_runs('obb')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)

    try:
        results = predict(model, images, conf=conf, verbose=False)
    except Exception as e:
        log(f'Error predicting: {e}', FILENAME)

    try:
        process_results(results)
    except Exception as e:
        log(f'Error processing results: {e}', FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop undetected images to only include the scale')
    parser.add_argument('-c', '--conf', type=float, default=0.8, help='Confidence threshold for the YOLO model')
    args = parser.parse_args()
    main(args.conf)
