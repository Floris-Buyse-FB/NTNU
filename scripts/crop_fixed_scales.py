from ultralytics import YOLO, settings
import os
import shutil
from IPython.display import display, Image
from IPython import display
import cv2
import matplotlib.pyplot as plt
from packages import log
display.clear_output()

# CONSTANTS
IMG_PATH_FIXED = './images/classification/fixed'
MODEL_PATH_NO_OBB = './models/BEST_no_obb.pt'
SAVE_DIR_FIXED = './images/cropped_scales/fixed'
RUNS_DIR = settings['runs_dir']
NO_OBB_PREDICT_PATH = os.path.join(RUNS_DIR, 'detect\\predict\\crops\\scale_fixed')

try:
    if not os.path.exists(SAVE_DIR_FIXED):
        os.makedirs(SAVE_DIR_FIXED)
except OSError as e:
    log(f'Error: {e}')
    print('Something went wrong, check the log file for more information')

def crop_scale_fixed(images: list):
    model_no_obb = YOLO(MODEL_PATH_NO_OBB)
    # remove old runs
    shutil.rmtree(os.path.join(RUNS_DIR, 'detect'), ignore_errors=True)
    # predict images
    results_fixed = model_no_obb(images, conf=0.8, save_crop=True)

    # move images to save dir
    predictions = os.listdir(NO_OBB_PREDICT_PATH)
    for pred in predictions:
        if pred.endswith('.jpg'):
            shutil.move(os.path.join(NO_OBB_PREDICT_PATH, pred), SAVE_DIR_FIXED)

    return results_fixed


def load_image(image_path, fig_size=(50, 50), grid=False, x_ticks=30, y_ticks=10, x_rotation=0, y_rotation=0, save=False, save_path=None):
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

        # Optionally, you can customize the ticks to match your image's scale
        plt.xticks(range(0, image_rgb.shape[1], x_ticks), rotation=x_rotation)  # Adjust the spacing as needed
        plt.yticks(range(0, image_rgb.shape[0], y_ticks), rotation=y_rotation)  # Adjust the spacing as needed

    if save:
        if save_path is None:
            plt.savefig(image_path.split('/')[-1], bbox_inches='tight', transparent=True)
            plt.close()
        else:
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
            plt.close()


try:
    all_images = os.listdir(IMG_PATH_FIXED)
    images = [os.path.join(IMG_PATH_FIXED, img) for img in all_images]

    results = crop_scale_fixed(images)
except Exception as e:
    log(f'Error cropping images: {e}')
    print('Something went wrong, check the log file for more information')

# rename crops
try:
    for idx, result in enumerate(results):
        if idx == 0:
            os.rename(os.path.join(SAVE_DIR_FIXED, 'labels.jpg'), os.path.join(SAVE_DIR_FIXED, result.path.split("\\")[-1].replace('.jpg', '') + '_scale_only.jpg'))
        else:
            os.rename(os.path.join(SAVE_DIR_FIXED, f'labels{idx+1}.jpg'), os.path.join(SAVE_DIR_FIXED, result.path.split("\\")[-1].replace('.jpg', '') + '_scale_only.jpg'))
except Exception as e:
    log(f'Error renaming crops: {e}')
    print('Something went wrong, check the log file for more information')

# adds grid to images
try:
    for idx, result in enumerate(results):
        load_image(result.path, grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save=True, save_path=os.path.join(SAVE_DIR_FIXED, result.path.split('\\')[-1].replace('.jpg', '_grid.jpg')))
except Exception as e:
    log(f'Error adding grid to images: {e}')
    print('Something went wrong, check the log file for more information')

try:
    for scale in os.listdir(SAVE_DIR_FIXED):
        if scale.endswith('_scale_only.jpg'):
            load_image(os.path.join(SAVE_DIR_FIXED, scale), grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save=True, save_path=os.path.join(SAVE_DIR_FIXED, scale.replace('_scale_only.jpg', '_scale_only_grid.jpg')))
except Exception as e:
    log(f'Error adding grid to cropped images: {e}')
    print('Something went wrong, check the log file for more information')

# remove old directories
try:
    shutil.rmtree(IMG_PATH_FIXED)
except Exception as e:
    log(f'Error removing old images: {e}')
    print('Something went wrong, check the log file for more information')