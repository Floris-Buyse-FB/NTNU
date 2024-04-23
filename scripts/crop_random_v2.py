import numpy as np
import argparse
import shutil
import torch
import cv2
import os

from packages import log, width_length_difference, check_directory, \
            get_image_paths, load_model, remove_old_runs, predict


# CONSTANTS
SAVE_DIR= './images/cropped_scales/random'
FILENAME = os.path.basename(__file__)

MODEL_PATH = './models/crop_random_scale.pt'


def process_results(results: list) -> None:
    """
    Process the results obtained from the model predictions.
    Crop the images based on the predicted bounding boxes and save them.
    
    Args:
        results (list): List of results obtained from the model predictions.
        
    Returns:
        None
    """
    # Loop over each list in the outer list
    for result_list in results:
        # Result is a list of results of a batch of images
        for result in result_list:
            # Result is a single result of a single image
            # Get the image path and name
            image_path = result.path
            image_name = os.path.basename(image_path)

            # Get the bounding box with the largest width-length difference
            # This also handles the case where there are multiple bounding boxes
            if len(result.obb.xyxyxyxy) > 1:
                # Uses the width_length_difference function to get the differences (see packages/helper.py)
                differences = torch.tensor([width_length_difference(box) for box in result.obb.xyxyxyxy])
                box_with_largest_difference_index = differences.argmax().item()
                bbox = result.obb.xyxyxyxy[box_with_largest_difference_index]
            else:
                bbox = result.obb.xyxyxyxy[0]
            
            # Open image
            image = cv2.imread(image_path)
            
            # Convert the bounding box to a numpy array
            quadrilateral = np.array(bbox.cpu() if torch.cuda.is_available() else bbox)
            
            # Get the bounding box coordinates in the format (x, y, w, h)
            x, y, w, h = cv2.boundingRect(quadrilateral)

            # Ensure the bounding box is within the image bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(x + w, image.shape[1])
            y2 = min(y + h, image.shape[0])

            # Crop the image based on the bounding box
            cropped_image = image[y:y2, x:x2]
            
            # Save the cropped image and move the original image to the save directory
            cv2.imwrite(os.path.join(SAVE_DIR, image_name.replace('.jpg', '_scale_only.jpg')), cropped_image)
            shutil.move(image_path, os.path.join(SAVE_DIR, image_name))


def main(conf: float) -> None:
    """
    Main function to execute the cropping process.
    
    Args:
        conf (float): Confidence threshold for the YOLO model.
        
    Returns:
        None
    """
    try:
        # Create the save directory if it does not exist
        check_directory(SAVE_DIR)
    except Exception as e:
        log(f'Error creating directory: {e}', FILENAME)

    try:
        # Load the model
        model = load_model(MODEL_PATH)
    except Exception as e:
        log(f'Error loading model: {e}', FILENAME)
    
    try:
        # retrieve the image paths (full path)
        images = get_image_paths(SAVE_DIR, ext='.jpg', neg_ext='_scale_only.jpg')
        
        # check if there are images to process
        images_check = [img for img in images if not os.path.exists(img.replace('.jpg', '_scale_only.jpg'))]

        if len(images_check) == 0:
            print('No images to process')
            return

    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)

    try:
        # remove old runs
        remove_old_runs('obb')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)

    try:
        # predict the images
        results = predict(model, images, conf=conf, verbose=False)
    except Exception as e:
        log(f'Error predicting: {e}', FILENAME)

    try:
        # process the results
        process_results(results)
    except Exception as e:
        log(f'Error processing results: {e}', FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop undetected images to only include the scale')
    parser.add_argument('-c', '--conf', type=float, default=0.8, help='Confidence threshold for the YOLO model')
    args = parser.parse_args()
    main(args.conf)
