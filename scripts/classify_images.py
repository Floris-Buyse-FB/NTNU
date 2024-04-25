import shutil
import os

from packages import log, check_directory, get_image_paths, \
    remove_old_runs, remove_old_directories, load_model, predict
from ultralytics import settings

# CONSTANTS
IMG_PATH = './images/gbif_images'
CLASSIFICATION_DIR = './images/classification'
CD_FIXED = os.path.join(CLASSIFICATION_DIR, 'fixed')
CD_RANDOM = os.path.join(CLASSIFICATION_DIR, 'random')
CD_NO_CLASS = os.path.join(CLASSIFICATION_DIR, 'no_class')
FILENAME = os.path.basename(__file__)
RUNS_DIR = '/home/floris/Projects/NTNU/models/runs'

# update settings of ultralytics (YOLOv8)
settings.update({'runs_dir': RUNS_DIR})

MODEL_PATH = './models/classify_scale_medium_v2.pt'


def move_images(results: list) -> None:
    """
    Move classified images to their respective directories based on the classification results.

    Args:
        results (list): A list of lists, where each inner list contains the classification results of a batch of images.
    
    Returns:
        None
    """
    # loop over each list in the outer list
    for result in results:
        # result is a list of results of a batch of images
        # loop over each result in the inner list
        for res in result:
            cls_size = res.boxes.cls.cpu().numpy().size # get the size of the list of detected classes
            cls_first = res.boxes.cls.cpu().numpy()[0] if cls_size else None # get the first detected class

            if cls_size == 0:
                dest_dir = CD_NO_CLASS
            elif cls_size == 1 and cls_first == 0:
                dest_dir = CD_FIXED
            elif cls_size == 1 and cls_first == 1:
                dest_dir = CD_RANDOM
            else:
                continue 
            
            # move the image to the destination directory
            shutil.move(res.path, os.path.join(dest_dir, os.path.basename(res.path)))


def main() -> None:
    """
    Main function to classify images and move them to their respective directories.
    
    Args:
        None
    
    Returns:
        None
    """
    try:
        # Create directories if they don't exist
        check_directory(CLASSIFICATION_DIR)
        check_directory(CD_FIXED)
        check_directory(CD_RANDOM)
        check_directory(CD_NO_CLASS)
    except Exception as e:
        log(f'Error creating directories: {e}', FILENAME)

    try:
        # Load yolo model
        model = load_model(MODEL_PATH)
    except Exception as e:
        log(f'Error loading model: {e}', FILENAME)
    
    try:
        # Function to get a list of image paths (full path)
        images = get_image_paths(IMG_PATH)
    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)
    
    try:
        # Yolo model saves predictions in a folder called 'detect'
        # remove_old_runs function removes the 'detect' folder
        # before running the model to avoid confusion with old runs
        remove_old_runs('detect')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)
    
    try:
        # Predict function returns a list of lists, where each inner list contains the results of a batch of images
        results = predict(model, images)
        move_images(results)
    except Exception as e:
        log(f'Error classifying images: {e}', FILENAME)
    
    try:
        # removes the /images/gbif_images directory (in this case)
        remove_old_directories([IMG_PATH])
    except Exception as e:
        log(f'Error removing old directories: {e}', FILENAME)

    log('Classification complete.', FILENAME)


if __name__ == '__main__':
    main()