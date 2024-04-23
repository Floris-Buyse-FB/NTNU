import numpy as np
import shutil
import torch
import cv2
import os

from ultralytics import settings
from packages import log, check_directory, get_image_paths, load_model, \
    remove_old_directories, remove_old_runs, predict, width_length_difference


# CONSTANTS
IMG_PATH = './images/classification/random'
SAVE_DIR = './images/cropped_scales/random'
FILENAME = os.path.basename(__file__)

MODEL_PATH= './models/crop_random_scale.pt'

# update settings of ultralytics (YOLOv8)
RUNS_DIR = settings['runs_dir']
PREDICT_PATH = os.path.join(RUNS_DIR, 'obb/predict')


def process_results(results: list) -> None:
    """
    Process the results obtained from the model predictions.
    Crop the images based on the predicted bounding boxes and save them in the specified directory.

    Args:
        results (list): List of results obtained from the model predictions.

    Returns:
        None
    """
    # loop over each list in the outer list
    for result_list in results:
        # result is a list of results of a batch of images
        for result in result_list:
            # result is a single result of a single image
            # get the image path and name
            image_path = result.path
            image_name = os.path.basename(image_path)

            # get the bounding box with the largest width-length difference
            # this also handles the case where there are multiple bounding boxes
            if len(result.obb.xyxyxyxy) > 1:
                # uses the width_length_difference function to get the differences (see packages/helper.py)
                differences = torch.tensor([width_length_difference(box) for box in result.obb.xyxyxyxy])
                box_with_largest_difference_index = differences.argmax().item()
                bbox = result.obb.xyxyxyxy[box_with_largest_difference_index]
            else:
                bbox = result.obb.xyxyxyxy[0]
            
            # open image
            image = cv2.imread(image_path)
            
            # convert the bounding box to a numpy array
            quadrilateral = np.array(bbox.cpu() if torch.cuda.is_available() else bbox)
            
            # get the bounding box coordinates in the format (x, y, w, h)
            x, y, w, h = cv2.boundingRect(quadrilateral)
            
            # crop the image based on the bounding box
            cropped_image = image[y:y+h, x:x+w]
            
            # save the cropped image and move the original image to the save directory
            cv2.imwrite(os.path.join(SAVE_DIR, image_name.replace('.jpg', '_scale_only.jpg')), cropped_image)
            shutil.move(image_path, os.path.join(SAVE_DIR, image_name))


def main() -> None:
    """
    Main function to execute the cropping of random scales.

    Args:
        None
    
    Returns:
        None
    """
    try:
        # check if the save directory exists, if not create it
        check_directory(SAVE_DIR)
    except Exception as e:
        log(f'Error creating directory: {e}', FILENAME)

    try:
        # load the model
        model = load_model(MODEL_PATH)
    except Exception as e:
        log(f'Error loading model: {e}', FILENAME)
    
    try:
        # retrieve a list of image paths (full path)
        images = get_image_paths(IMG_PATH)
    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)

    try:
        # remove old runs
        remove_old_runs('obb')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)

    try:
        # predict the images
        results = predict(model, images, conf=0.7, verbose=False)
    except Exception as e:
        log(f'Error predicting images: {e}', FILENAME)

    try:
        # process the results
        process_results(results)
    except Exception as e:
        log(f'Error processing results: {e}', FILENAME)
        
    try:
        # remove old directories
        remove_old_directories([IMG_PATH, PREDICT_PATH])
    except Exception as e:
        log(f'Error removing directories: {e}', FILENAME)


if __name__ == '__main__':
    main()