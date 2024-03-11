from ultralytics import YOLO, settings
import os
import shutil
from IPython.display import display, Image
from IPython import display
display.clear_output()


IMG_PATH_FIXED = './images/scale_fixed'
IMG_PATH_RANDOM = './images/scale_random'
MODEL_PATH_OBB = './models/yolov8n_obb_100epochs.pt'
MODEL_PATH_NO_OBB = './models/no_obb_best_05_03_24.pt'
CLASSIFICATION_DIR = './images/classification'
settings.update({'runs_dir': rf'C:\Users\buyse\Workspace\NTNU\models\runs'})

model_obb = YOLO(MODEL_PATH_OBB)
model_no_obb = YOLO(MODEL_PATH_NO_OBB)


all_images_fixed = sorted(os.listdir(IMG_PATH_FIXED))
all_images_random = sorted(os.listdir(IMG_PATH_RANDOM))

all_images_fixed = [os.path.join(IMG_PATH_FIXED, img) for img in all_images_fixed]
all_images_random = [os.path.join(IMG_PATH_RANDOM, img) for img in all_images_random]

shutil.rmtree(rf'C:\Users\buyse\Workspace\NTNU\models\runs\obb', ignore_errors=True)

combined_images = all_images_fixed + all_images_random

results = model_obb(combined_images, save=True, save_txt=True, save_conf=True, conf=0.8)

with open('./models/runs/obb/predict/labels.txt', 'r') as f:
    labels = f.read().splitlines()

image_dict = {}

# TODO: Here, the order of the results and labels is not guaranteed to be the same. -> solve this

for idx, result in enumerate(results):

    try:
        coords = result.obb.xyxyxyxyn[0].flatten().tolist()
        coords = [round(float(item), 4) for item in coords]
    except:
        continue

    try:
        label_coords = labels[idx].split(' ')[1:-1]
        label_coords = [round(float(item), 4) for item in label_coords]
    except:
        continue

    if coords == label_coords:
        path_clean = result.path.split('/', 2)[-1].replace('\\', '/')
        if path_clean not in image_dict:
            image_dict[path_clean] = []
        image_dict[path_clean].append(labels[idx].split(' ')[0])

for key, value in image_dict.items():
    if len(value) == 1:
        if value[0] == '0':
            shutil.move(os.path.join('./images', key), os.path.join(CLASSIFICATION_DIR + '/fixed', key.split('/')[-1]))
        elif value[0] == '1':
            shutil.move(os.path.join('./images', key), os.path.join(CLASSIFICATION_DIR + '/random', key.split('/')[-1]))
        else:
            shutil.move(os.path.join('./images', key), os.path.join(CLASSIFICATION_DIR + '/no_class', key.split('/')[-1]))