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

if not os.path.exists(CLASSIFICATION_DIR):
    os.makedirs(CLASSIFICATION_DIR)
    os.makedirs(os.path.join(CLASSIFICATION_DIR, 'fixed'))
    os.makedirs(os.path.join(CLASSIFICATION_DIR, 'random'))
    os.makedirs(os.path.join(CLASSIFICATION_DIR, 'no_class'))

model_obb = YOLO(MODEL_PATH_OBB)
model_no_obb = YOLO(MODEL_PATH_NO_OBB)


all_images_fixed = sorted(os.listdir(IMG_PATH_FIXED))
all_images_random = sorted(os.listdir(IMG_PATH_RANDOM))

all_images_fixed = [os.path.join(IMG_PATH_FIXED, img) for img in all_images_fixed]
all_images_random = [os.path.join(IMG_PATH_RANDOM, img) for img in all_images_random]

combined_images = all_images_fixed + all_images_random

shutil.rmtree(rf'C:\Users\buyse\Workspace\NTNU\models\runs\obb', ignore_errors=True)

results = model_obb(combined_images, conf=0.8)

for result in results:
    class_label = result.obb.cls # tensor with 1 item if detected, no item if not detected
    path = result.path

    if len(class_label) > 0:
        if class_label[0] == 0:
            shutil.move(path, os.path.join(CLASSIFICATION_DIR, 'fixed', os.path.basename(path)))
        elif class_label[0] == 1:
            shutil.move(path, os.path.join(CLASSIFICATION_DIR, 'random', os.path.basename(path)))
    else:
        shutil.move(path, os.path.join(CLASSIFICATION_DIR, 'no_class', os.path.basename(path)))