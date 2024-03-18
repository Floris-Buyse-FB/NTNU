from ultralytics import YOLO
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from packages import log
import torch
import argparse

# CONSTANTS
SAVE_DIR_RANDOM = './images/cropped_scales/random'
MODEL_PATH_OBB = './models/BEST_m_obb.pt'


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


def predict_and_crop_image(image_path, conf):
    model_obb = YOLO(MODEL_PATH_OBB)
    result = model_obb(image_path, conf=conf)
    if len(result[0].obb.xyxyxyxy) > 1:
        differences = torch.tensor([width_length_difference(box) for box in result[0].obb.xyxyxyxy.cpu()])
        box_with_largest_difference_index = differences.argmax().item()
        bbox = result[0].obb.xyxyxyxy[box_with_largest_difference_index].cpu().numpy()
    else:
        bbox = result[0].obb.xyxyxyxy[0].cpu().numpy()
    image = cv2.imread(image_path)
    quadrilateral = np.array(bbox)
    x, y, w, h = cv2.boundingRect(quadrilateral)
    
    # Ensure the bounding box is within the image bounds
    x = max(0, x)
    y = max(0, y)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])
    
    cropped_image = image[y:y2, x:x2]
    image_name = os.path.basename(image_path)
    cropped_image_path = os.path.join(SAVE_DIR_RANDOM, image_name.replace('.jpg', '_scale_only.jpg'))
    cv2.imwrite(cropped_image_path, cropped_image)
    print(f'Saved cropped image to {cropped_image_path}')
    load_image(cropped_image_path, grid=True, x_ticks=120, y_ticks=10, x_rotation=90, y_rotation=0, save=True, save_path=os.path.join(SAVE_DIR_RANDOM, image_name.replace('.jpg', '_scale_only_grid.jpg')))


def main(conf):
    # Get a list of all image files in the SAVE_DIR_RANDOM directory
    image_files = [f for f in os.listdir(SAVE_DIR_RANDOM) if f.endswith('.jpg') and '_scale_only' not in f and '_grid' not in f]
    image_files = [f for f in image_files if not os.path.exists(os.path.join(SAVE_DIR_RANDOM, f.replace('.jpg', '_scale_only.jpg')))]

    if len(image_files) == 0:
        print('No images to process')
        return

    for image_file in image_files:
        image_path = os.path.join(SAVE_DIR_RANDOM, image_file)
        cropped_image_path = os.path.join(SAVE_DIR_RANDOM, image_file.replace('.jpg', '_scale_only.jpg'))
        grid_image_path = os.path.join(SAVE_DIR_RANDOM, image_file.replace('.jpg', '_grid.jpg'))

        if not os.path.exists(cropped_image_path) or not os.path.exists(grid_image_path):
            try:
                predict_and_crop_image(image_path, conf)
            except Exception as e:
                log(f'Error processing image {image_file}: {e}')
                print(f'Something went wrong with image {image_file}, check the log file for more information')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop undetected images to only include the scale')
    parser.add_argument('-c', '--conf', type=float, default=0.8, help='Confidence threshold for the YOLO model')
    args = parser.parse_args()
    main(args.conf)
