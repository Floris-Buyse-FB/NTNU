from packages import log, check_directory, remove_old_runs, load_model, predict, get_image_paths, remove_old_directories
from ultralytics import settings
import shutil
import os


# CONSTANTS
IMG_PATH = './images/classification/fixed'
SAVE_DIR = './images/cropped_scales/fixed'
FILENAME = os.path.basename(__file__)

MODEL_PATH = './models/crop_fixed_scale.pt'

settings.update({'runs_dir': rf'/home/floris/Projects/NTNU/models/runs'})
RUNS_DIR = settings['runs_dir']
PREDICT_PATH = os.path.join(RUNS_DIR, 'detect/predict/crops/scale_fixed')


def process_results(results, name_label, cropped_scales):
    if name_label:
        for idx, result in enumerate(results):
            new_name = result.path.split('/')[-1].replace('.jpg', '_scale_only.jpg')
            if idx == 0:
                os.rename(os.path.join(PREDICT_PATH, 'labels.jpg'), os.path.join(PREDICT_PATH, new_name))
                shutil.move(os.path.join(PREDICT_PATH, new_name), SAVE_DIR)
            else:
                os.rename(os.path.join(PREDICT_PATH, f'labels{idx+1}.jpg'), os.path.join(PREDICT_PATH, new_name))
                shutil.move(os.path.join(PREDICT_PATH, new_name), SAVE_DIR)
    else:
        for pred in cropped_scales:
            if pred.endswith('.jpg') and 'label' not in pred:
                image_name = pred.replace('.jpg', '_scale_only.jpg')
                os.rename(os.path.join(PREDICT_PATH, pred), os.path.join(PREDICT_PATH, image_name))
                shutil.move(os.path.join(PREDICT_PATH, image_name), SAVE_DIR)


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
        remove_old_runs('detect')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)
    
    try:
        results = predict(model, images, conf=0.8, save_crop=True, verbose=True)
    except Exception as e:
        log(f'Error predicting: {e}', FILENAME)

    try:
        cropped_scales = os.listdir(PREDICT_PATH)
        name_label = True if 'labels.jpg' in cropped_scales else False
    except Exception as e:
        log(f'Error checking if the cropped images are called label or their gbifID: {e}', FILENAME)

    try:
        process_results(results, name_label, cropped_scales)
    except Exception as e:
        log(f'Error processing results: {e}', FILENAME)
    
    try:
        for img in images:
            shutil.move(img, os.path.join(SAVE_DIR, os.path.basename(img)))
    except Exception as e:
        log(f'Error moving original images: {e}', FILENAME)
    
    try:
        remove_old_directories([IMG_PATH, PREDICT_PATH])
    except Exception as e:
        log(f'Error removing old directories: {e}', FILENAME)


if __name__ == '__main__':
    main()