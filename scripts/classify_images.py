from packages import log, check_directory, get_image_paths, \
                     remove_old_runs, remove_old_directories, \
                     load_model, predict
import shutil
import os


# CONSTANTS
IMG_PATH = './images/gbif_images'
CLASSIFICATION_DIR = './images/classification'
CD_FIXED = os.path.join(CLASSIFICATION_DIR, 'fixed')
CD_RANDOM = os.path.join(CLASSIFICATION_DIR, 'random')
CD_NO_CLASS = os.path.join(CLASSIFICATION_DIR, 'no_class')
FILENAME = os.path.basename(__file__)

MODEL_PATH = './models/classify_scale_medium_v2.pt'


def move_images(results):
    for result in results:
        for res in result:
            cls_size = res.boxes.cls.cpu().numpy().size
            cls_first = res.boxes.cls.cpu().numpy()[0] if cls_size else None

            if cls_size == 0:
                dest_dir = CD_NO_CLASS
            elif cls_size == 1 and cls_first == 0:
                dest_dir = CD_FIXED
            elif cls_size == 1 and cls_first == 1:
                dest_dir = CD_RANDOM
            else:
                continue 

            shutil.move(res.path, os.path.join(dest_dir, os.path.basename(res.path)))


def main():

    try:
        check_directory(CLASSIFICATION_DIR)
        check_directory(CD_FIXED)
        check_directory(CD_RANDOM)
        check_directory(CD_NO_CLASS)
    except Exception as e:
        log(f'Error creating directories: {e}', FILENAME)

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        log(f'Error loading model: {e}', FILENAME)
    
    try:
        images = get_image_paths(IMG_PATH)
    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)
    
    try:
        remove_old_runs('detect')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)
    
    try:
        results = predict(model, images) # results is a list of lists, each list contains the results of a batch of images
        move_images(results)
    except Exception as e:
        log(f'Error classifying images: {e}', FILENAME)
    
    try:
        remove_old_directories([IMG_PATH])
    except Exception as e:
        log(f'Error removing old directories: {e}', FILENAME)

    log('Classification complete.', FILENAME)


if __name__ == '__main__':
    main()