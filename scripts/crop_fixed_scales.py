from ultralytics import YOLO, settings
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from packages import log

# CONSTANTS
IMG_PATH_FIXED = './images/classification/fixed'
MODEL_PATH_NO_OBB = './models/crop_fixed_scale.pt'
SAVE_DIR_FIXED = './images/cropped_scales/fixed'
settings.update({'runs_dir': rf'C:\Users\buyse\Workspace\NTNU\models\runs'})
RUNS_DIR = settings['runs_dir']
NO_OBB_PREDICT_PATH = os.path.join(RUNS_DIR, 'detect/predict/crops/scale_fixed')

def create_directory(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        log(f'Error: {e}')
        print('Something went wrong, check the log file for more information')

def crop_scale_fixed(images: list):
    model_no_obb = YOLO(MODEL_PATH_NO_OBB)
    # remove old runs
    shutil.rmtree(os.path.join(RUNS_DIR, 'detect/predict'), ignore_errors=True)
    # predict images
    results_fixed = model_no_obb(images, conf=0.8, save_crop=True, verbose=True)
    # rename and move images to save dir
    predictions = os.listdir(NO_OBB_PREDICT_PATH)
    name_label = True if 'labels.jpg' in predictions else False
    if name_label:
        for idx, result in enumerate(results_fixed):
                new_name = result.path.split("/")[-1].replace('.jpg', '_scale_only.jpg')
                if idx == 0:
                    os.rename(os.path.join(NO_OBB_PREDICT_PATH, 'labels.jpg'), os.path.join(NO_OBB_PREDICT_PATH, new_name))
                    shutil.move(os.path.join(NO_OBB_PREDICT_PATH, new_name), SAVE_DIR_FIXED)
                else:
                    os.rename(os.path.join(NO_OBB_PREDICT_PATH, f'labels{idx+1}.jpg'), os.path.join(NO_OBB_PREDICT_PATH, new_name))
                    shutil.move(os.path.join(NO_OBB_PREDICT_PATH, new_name), SAVE_DIR_FIXED)
    else:
        for pred in predictions:
            if pred.endswith('.jpg') and 'label' not in pred:
                image_name = pred.replace('.jpg', '_scale_only.jpg')
                os.rename(os.path.join(NO_OBB_PREDICT_PATH, pred), os.path.join(NO_OBB_PREDICT_PATH, image_name))
                shutil.move(os.path.join(NO_OBB_PREDICT_PATH, image_name), SAVE_DIR_FIXED)
            
    return results_fixed

def load_and_save_image(image_path, fig_size=(50, 50), grid=False, x_ticks=30, y_ticks=10, x_rotation=0, y_rotation=0, save_path=None):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB (matplotlib uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a figure with a specific size
    plt.figure(figsize=fig_size)
    # Show the image
    plt.imshow(image_rgb)
    if grid:
        # Add grid lines to the image for easier measurement
        plt.grid(color='r', linestyle='-', linewidth=0.5)
        # Customize the ticks to match your image's scale
        plt.xticks(range(0, image_rgb.shape[1], x_ticks), rotation=x_rotation)
        plt.yticks(range(0, image_rgb.shape[0], y_ticks), rotation=y_rotation)
    if save_path is None:
        save_path = image_path.split('/')[-1]
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close()

def process_images():
    create_directory(SAVE_DIR_FIXED)
    
    try:
        all_images = os.listdir(IMG_PATH_FIXED)
        images = [os.path.join(IMG_PATH_FIXED, img).replace('\\', '/') for img in all_images]
        if len(images) > 50:
            # predict in batches of 50
            for i in range(0, len(images), 50):
                batch = images[i:i+50]
                crop_scale_fixed(batch)
        else:
            crop_scale_fixed(images)
    except Exception as e:
        log(f'Error cropping images: {e}')
        print('Something went wrong, check the log file for more information')
        return
    
    ## Add grid to images
    # try:
    #     for result in results:
    #         save_path = os.path.join(SAVE_DIR_FIXED, result.path.split('/')[-1].replace('.jpg', '_grid.jpg'))
    #         load_and_save_image(result.path, grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save_path=save_path)
    # except Exception as e:
    #     log(f'Error adding grid to images: {e}')
    #     print('Something went wrong, check the log file for more information')
    
    # Move original images to save dir
    try:
        for img in images:
            shutil.move(img, SAVE_DIR_FIXED)
    except Exception as e:
        log(f'Error moving original images: {e}')
        print('Something went wrong, check the log file for more information')
    
    # # Add grid to cropped images
    # try:
    #     for scale in os.listdir(SAVE_DIR_FIXED):
    #         if scale.endswith('_scale_only.jpg'):
    #             save_path = os.path.join(SAVE_DIR_FIXED, scale.replace('_scale_only.jpg', '_scale_only_grid.jpg'))
    #             load_and_save_image(os.path.join(SAVE_DIR_FIXED, scale), grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save_path=save_path)
    # except Exception as e:
    #     log(f'Error adding grid to cropped images: {e}')
    #     print('Something went wrong, check the log file for more information')
    
    try:
        shutil.rmtree(IMG_PATH_FIXED)
    except Exception as e:
        log(f'Error removing old images: {e}')
        print('Something went wrong, check the log file for more information')

if __name__ == '__main__':
    process_images()