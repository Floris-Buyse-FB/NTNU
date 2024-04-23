import shutil
import os

from ultralytics import settings
from packages import log, check_directory, remove_old_runs, load_model, \
                    predict, get_image_paths, remove_old_directories


# CONSTANTS
IMG_PATH = './images/classification/fixed'
SAVE_DIR = './images/cropped_scales/fixed'
FILENAME = os.path.basename(__file__)

MODEL_PATH = './models/crop_fixed_scale.pt'

# update settings of ultralytics (YOLOv8)
settings.update({'runs_dir': rf'/home/floris/Projects/NTNU/models/runs'})
RUNS_DIR = settings['runs_dir']
PREDICT_PATH = os.path.join(RUNS_DIR, 'detect/predict/crops/scale_fixed')


def process_results(results: list, name_label: bool, cropped_scales: list) -> None:
    """
    Process the results of the prediction and move the cropped scale images to the save directory.
    name_label is True if the cropped images are named 'labels.jpg' (YOLO sometimes names the cropped images 'labelsX.jpg')
    There is no clear pattern in the naming of the cropped images, so we need to check if the cropped images are named 'labels.jpg' or not.

    Args:
        results (list): List of prediction results.
        name_label (bool): Flag indicating if the cropped images are named 'labels.jpg'.
        cropped_scales (list): List of cropped scale images.

    Returns:
        None
    """
    if name_label:
        for idx, result in enumerate(results):
            new_name = result.path.split('/')[-1].replace('.jpg', '_scale_only.jpg') # result.path is the full path of the original image
            if idx == 0:
                # if the idx is 0, the cropped image is named 'labels.jpg' without a number
                # rename and move the cropped image to the save directory
                os.rename(os.path.join(PREDICT_PATH, 'labels.jpg'), os.path.join(PREDICT_PATH, new_name))
                shutil.move(os.path.join(PREDICT_PATH, new_name), SAVE_DIR)
            else:
                # if the idx is not 0, the cropped image is named 'labelsX.jpg' with X being the idx
                # rename and move the cropped image to the save directory
                os.rename(os.path.join(PREDICT_PATH, f'labels{idx+1}.jpg'), os.path.join(PREDICT_PATH, new_name))
                shutil.move(os.path.join(PREDICT_PATH, new_name), SAVE_DIR)
    else:
        # if the cropped images names are kept as their gbifID
        for pred in cropped_scales:
            if pred.endswith('.jpg') and 'label' not in pred: # dubbel check if the file is an image and not a label
                # rename and move the cropped image to the save directory
                image_name = pred.replace('.jpg', '_scale_only.jpg')
                os.rename(os.path.join(PREDICT_PATH, pred), os.path.join(PREDICT_PATH, image_name))
                shutil.move(os.path.join(PREDICT_PATH, image_name), SAVE_DIR)


def main() -> None:
    """
    Main function to execute the cropping of fixed scales.
    
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
        remove_old_runs('detect')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)
    
    try:
        # predict the images, save_crop is set to True to save the cropped images
        # verbose set to True as it sometimes doesn't save the cropped images when set to False
        results = predict(model, images, conf=0.8, save_crop=True, verbose=True)
    except Exception as e:
        log(f'Error predicting: {e}', FILENAME)

    try:
        # check if the cropped images are named 'labels.jpg' or their gbifID
        cropped_scales = os.listdir(PREDICT_PATH)
        name_label = True if 'labels.jpg' in cropped_scales else False
    except Exception as e:
        log(f'Error checking if the cropped images are called label or their gbifID: {e}', FILENAME)

    try:
        # process results
        process_results(results, name_label, cropped_scales)
    except Exception as e:
        log(f'Error processing results: {e}', FILENAME)
    
    try:
        # move the original images to the save directory
        for img in images:
            shutil.move(img, os.path.join(SAVE_DIR, os.path.basename(img)))
    except Exception as e:
        log(f'Error moving original images: {e}', FILENAME)
    
    try:
        # removes the /images/classification/fixed directory and the
        # /runs/detect/predict/crops/scale_fixed directory (in this case)
        remove_old_directories([IMG_PATH, PREDICT_PATH])
    except Exception as e:
        log(f'Error removing old directories: {e}', FILENAME)


if __name__ == '__main__':
    main()