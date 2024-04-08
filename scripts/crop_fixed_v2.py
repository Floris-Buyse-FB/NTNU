from packages import log, remove_old_runs, load_model, predict, get_image_paths, remove_old_directories
from ultralytics import settings
import argparse
import shutil
import os

SAVE_DIR= './images/cropped_scales/fixed'
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
                
def main(conf=0.8):
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
        
        print(f'Images to process: {len(images_check)}')
        
    except Exception as e:
        log(f'Error getting image paths: {e}', FILENAME)

    try:
        remove_old_runs('detect')
    except Exception as e:
        log(f'Error removing old runs: {e}', FILENAME)
    
    try:
        results = predict(model, images_check, conf=conf, save_crop=True, verbose=True)
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
        remove_old_directories([PREDICT_PATH])
    except Exception as e:
        log(f'Error removing old directories: {e}', FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=float, default=0.8, help='Confidence threshold')
    args = parser.parse_args()
    main(args.conf)