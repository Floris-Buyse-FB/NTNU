import os
import cv2
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO, settings


def check_directory(path):
    "Check if a directory exists, if not, create it."
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_data(url, sep):
    "Read data from a csv file and return a pandas dataframe."
   
    data = pd.read_csv(url, sep=sep)
    return data


def number_to_letter(number):
    if 1 <= number <= 26:
        return chr(number + 96)
    else:
        return "Number out of range"


def get_image_paths(img_path, sort=True, ext=None, neg_ext=None):
    file_list = os.listdir(img_path)
    if sort:
        file_list = sorted(file_list)

    if ext is not None and neg_ext is None:
        filtered_files = [f for f in file_list if f.endswith(ext)]
    elif ext is None and neg_ext is not None:
        filtered_files = [f for f in file_list if not f.endswith(neg_ext)]
    elif ext is not None and neg_ext is not None:
        filtered_files = [f for f in file_list if f.endswith(ext) and not f.endswith(neg_ext)]
    else:
        filtered_files = file_list

    return [os.path.join(img_path, img) for img in filtered_files]


def remove_old_runs(mode):
    shutil.rmtree(os.path.join(settings['runs_dir'], mode), ignore_errors=True)


def remove_old_directories(dirs: list):
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors=True)


def load_model(path):
    return YOLO(path)


def predict(model, data: list, batch=22, conf=0.6, iou=0.7, half=True, verbose=True, classes=None, save=False, retina_masks=False, save_crop=False, \
            save_txt=False, show_labels=False, show_conf=False, show_boxes=True, imgsz=640) -> list:
    
    if len(data) > batch:
        results = []
        for i in range(0, len(data), batch):
            res = model(data[i:i+batch], conf=conf, iou=iou, half=half, verbose=verbose, classes=classes, save=save, retina_masks=retina_masks, \
                                 save_crop=save_crop, save_txt=save_txt, show_labels=show_labels, show_conf=show_conf, show_boxes=show_boxes, imgsz=imgsz)
            results.append(res)
        return results
    else:
        res = model(data, conf=conf, iou=iou, half=half, verbose=verbose, classes=classes, save=save, retina_masks=retina_masks, \
                 save_crop=save_crop, save_txt=save_txt, show_labels=show_labels, show_conf=show_conf, show_boxes=show_boxes, imgsz=imgsz)
        return [res]


def load_and_save_image(image_path, fig_size=(50, 50), grid=False, x_ticks=30, y_ticks=10, x_rotation=0, y_rotation=0, save_path=None):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=fig_size)
    plt.imshow(image_rgb)
    if grid:
        plt.grid(color='r', linestyle='-', linewidth=0.5)
        plt.xticks(range(0, image_rgb.shape[1], x_ticks), rotation=x_rotation)
        plt.yticks(range(0, image_rgb.shape[0], y_ticks), rotation=y_rotation)
    if save_path is None:
        pass
    else:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close()