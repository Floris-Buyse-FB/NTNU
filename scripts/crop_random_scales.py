from ultralytics import YOLO, settings
import os
import shutil
from IPython.display import display, Image
from IPython import display
import cv2
import matplotlib.pyplot as plt
display.clear_output()

# CONSTANTS
IMG_PATH_RANDOM = './images/classification/random'
MODEL_PATH_OBB = './models/yolov8n_obb_100epochs.pt'
SAVE_DIR_RANDOM = './images/cropped_scales/random'
RUNS_DIR = settings['runs_dir']
OBB_PREDICT_PATH = os.path.join(RUNS_DIR, 'obb\\predict')

if not os.path.exists(SAVE_DIR_RANDOM):
    os.makedirs(SAVE_DIR_RANDOM)


def crop_scale_fixed(images: list):
    model_obb = YOLO(MODEL_PATH_OBB)
    # remove old runs
    shutil.rmtree(os.path.join(RUNS_DIR, 'obb'), ignore_errors=True)
    # predict images
    results_fixed = model_obb(images, conf=0.8)

    # move images to save dir
    predictions = os.listdir(OBB_PREDICT_PATH)
    for pred in predictions:
        if pred.endswith('.jpg'):
            shutil.move(os.path.join(OBB_PREDICT_PATH, pred), SAVE_DIR_RANDOM)

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


all_images = os.listdir(IMG_PATH_RANDOM)
images = [os.path.join(IMG_PATH_RANDOM, img) for img in all_images]

results = crop_scale_fixed(images)