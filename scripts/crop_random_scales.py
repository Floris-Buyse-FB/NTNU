from ultralytics import YOLO, settings
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from packages import log
import torch

# CONSTANTS
IMG_PATH_RANDOM = './images/classification/random'
MODEL_PATH_OBB = './models/BEST_m_obb.pt'
SAVE_DIR_RANDOM = './images/cropped_scales/random'
RUNS_DIR = settings['runs_dir']
OBB_PREDICT_PATH = os.path.join(RUNS_DIR, 'obb/predict')

def create_directory(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        log(f'Error: {e}')
        print('Something went wrong, check the log file for more information')

def crop_scale_random(images: list):
    model_obb = YOLO(MODEL_PATH_OBB)
    # remove old runs
    shutil.rmtree(os.path.join(RUNS_DIR, 'obb'), ignore_errors=True)
    # predict images
    results_fixed = model_obb(images, conf=0.7)
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
        save_path = save_path or image_path.split('/')[-1]
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.close()

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return ((p1 - p2) ** 2).sum().sqrt()

def width_length_difference(box):
    """Calculate the width and length of a bounding box and return their difference."""
    # Calculate lengths of all sides: AB, BC, CD, DA
    sides = torch.tensor([
        distance(box[0], box[1]),
        distance(box[1], box[2]),
        distance(box[2], box[3]),
        distance(box[3], box[0])
    ])
    # The length is the maximum of the four sides, and width is the second maximum (considering a rotated box)
    length, width = sides.topk(2).values
    # Return the absolute difference between length and width
    return abs(length - width)

def process_results(results):
    for result in results:
        try:
            image_path = result.path.replace('\\', '/')
            image_name = image_path.split('/')[-1]
            if len(result.obb.xyxyxyxy) > 1:
                differences = torch.tensor([width_length_difference(box) for box in result.obb.xyxyxyxy])
                box_with_largest_difference_index = differences.argmax().item()
                bbox = result.obb.xyxyxyxy[box_with_largest_difference_index]
            else:
                bbox = result.obb.xyxyxyxy[0]

            # Load the image
            image = cv2.imread(image_path)
            # Define the quadrilateral
            quadrilateral = np.array(bbox.cpu() if torch.cuda.is_available() else bbox)
            # Compute axis aligned bounding box of the quadrilateral
            x, y, w, h = cv2.boundingRect(quadrilateral)
            # Crop the image
            cropped_image = image[y:y + h, x:x + w]
            # Save the result
            cv2.imwrite(os.path.join(SAVE_DIR_RANDOM, image_name.replace('.jpg', '_scale_only.jpg')), cropped_image)
            # Move original image
            shutil.move(image_path, os.path.join(SAVE_DIR_RANDOM, image_name))
            # Add grid to the image
            load_image(os.path.join(SAVE_DIR_RANDOM, image_name), grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save=True, save_path=os.path.join(SAVE_DIR_RANDOM, image_name.replace('.jpg', '_grid.jpg')))
            # Add grid to the cropped image
            try:
                load_image(os.path.join(SAVE_DIR_RANDOM, image_name.replace('.jpg', '_scale_only.jpg')), grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save=True, save_path=os.path.join(SAVE_DIR_RANDOM, image_name.replace('.jpg', '_scale_only_grid.jpg')))
            except:
                pass
        except Exception as e:
            log(f'Error cropping images: {e}')
            print('Something went wrong, check the log file for more information')

def main():
    create_directory(SAVE_DIR_RANDOM)
    try:
        all_images = os.listdir(IMG_PATH_RANDOM)
        images = [os.path.join(IMG_PATH_RANDOM, img) for img in all_images]
        results = crop_scale_random(images)
        process_results(results)
    except Exception as e:
        log(f'Error cropping images: {e}')
        print('Something went wrong, check the log file for more information')
    # Remove old directory
    try:
        shutil.rmtree(OBB_PREDICT_PATH, ignore_errors=True)
        shutil.rmtree(IMG_PATH_RANDOM, ignore_errors=True)
    except Exception as e:
        log(f'Error removing old images: {e}')
        print('Something went wrong, check the log file for more information')

if __name__ == '__main__':
    main()