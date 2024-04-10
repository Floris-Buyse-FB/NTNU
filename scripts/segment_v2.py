import sys
sys.path.append('..')

from ultralytics import YOLO, settings
from packages import check_directory
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import shutil
import torch
import json
import cv2
import os

torch.cuda.empty_cache()
CUDA = torch.cuda.is_available()
print("CUDA is available:", CUDA)

MODEL_PATH_SEGMENT = '/home/floris/Projects/NTNU/models/plant_segmentation_v2.pt'

IMG_PATH_FIXED = '/home/floris/Projects/NTNU/images/cropped_scales/fixed'
IMG_PATH_RANDOM = '/home/floris/Projects/NTNU/images/cropped_scales/random'
SAVE_PATH_RANDOM = '/home/floris/Projects/NTNU/data/processed/random'
SAVE_PATH_FIXED = '/home/floris/Projects/NTNU/data/processed/fixed'

DEVICE = "cuda" if CUDA else "cpu"
settings.update({'runs_dir': rf'/home/floris/Projects/NTNU/models/runs'})

model_seg = YOLO(MODEL_PATH_SEGMENT)

def measure_scale_random(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _max = 0
    _max_idx = 0
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if h > w:
            if abs(w - h) < 10 and w >= 0.18 * image.shape[0]:
                _max_idx = idx if w > _max else _max_idx
                _max = max(w, _max)
        else:
            if abs(w - h) < 10 and w >= 0.18 * image.shape[1]:
                _max_idx = idx if h > _max else _max_idx
                _max = max(h, _max)

    _, _, w, h = cv2.boundingRect(contours[_max_idx])

    px_per_cm_w = w / 2
    px_per_cm_h = h / 2
    px_per_cm = (px_per_cm_w + px_per_cm_h) / 2

    return px_per_cm

def measure_scale_fixed_via_colorboard(image_path, box_width_mm=6.2):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    mask = cv2.inRange(image_rgb, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, w, _ = cv2.boundingRect(largest_contour)
    pixels_per_cm = w / (box_width_mm / 10)
    return pixels_per_cm

def transform_px_to_cm(box, px_per_cm):
    """
    Function to transform the width and height of a box from pixels to cm
    """
    try:
        w = np.abs((box[2] - box[0]).cpu())
        h = np.abs((box[3] - box[1]).cpu())
    except:
        w = np.abs(box[2] - box[0])
        h = np.abs(box[3] - box[1])
    return w / px_per_cm, h / px_per_cm

def get_masked_image(image, mask):
    """
    Apply a mask to an image with transparency
    """
    # Remove single-dimensional entry from the shape of the mask
    mask_squeezed = np.squeeze(mask)  # This should change mask shape to (5831, 3391)
    # Generate an alpha channel where mask is True (255) and False (0)
    alpha_channel = np.where(mask_squeezed, 255, 0).astype(np.uint8)
    # Ensure alpha channel is correctly shaped [H, W] -> [H, W, 1]
    alpha_channel_shaped = np.expand_dims(alpha_channel, axis=-1)

    # print("Image shape:", image.size)
    # print("Alpha channel shape:", alpha_channel_shaped.shape)

    # Concatenate the alpha channel with the image to create an RGBA image
    rgba_image = np.concatenate((image, alpha_channel_shaped), axis=-1)
    return rgba_image

def get_cropped_image(image, box):
    """
    Crop an image with a given box
    """
    if isinstance(box, list):
        box = box[0]
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy() 
    else:
        box = np.array(box) 

    if len(box.shape) > 1:
        box = box[0]
    x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
    return image[y:y+h, x:x+w]

def apply_crop_mask(image, mask, box):
    """
    Apply a mask to an image and crop the image with a given box
    Returns a list of tuples with the masked image and the cropped image
    """
    images = []
    if len(np.array(mask).shape) == 3:
        for i, m in enumerate(mask):
            m_img = get_masked_image(image, m)
            b = box[i].cpu().numpy() if type(box) == torch.Tensor else box[i]
            crop_img = get_cropped_image(m_img, b)
            images.append((m_img, crop_img))
    else:
        m_img = get_masked_image(image, mask)
        b = box.cpu().numpy() if type(box) == torch.Tensor else box
        crop_img = get_cropped_image(m_img, b)
        images.append((m_img, crop_img))
    return images

def find_dominant_color(image, k=5):
    # Convert image to numpy array
    img_array = np.array(image)
    # Reshape it to a list of RGB values
    img_vector = img_array.reshape((-1, 3))
    # Run k-means on the pixel colors (fit only on a subsample to speed up)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_vector[::50])
    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    # Create a mask for pixels within a certain distance from the dominant color
    distances = np.sqrt(np.sum((img_vector - dominant_color) ** 2, axis=1))
    mask = distances < np.std(distances)
    # Turn the dominant color range to white
    img_vector[mask] = [255, 255, 255]
    result_img_array = img_vector.reshape(img_array.shape)
    # turn image back to PIL
    result_img = Image.fromarray(result_img_array.astype(np.uint8))
    return dominant_color, result_img

def calculate_mask_area(masked_pixels, pixels_per_cm):
    area_square_cm = masked_pixels / (pixels_per_cm ** 2)
    return area_square_cm

def get_images(path, range_left=0, range_right=None):
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        return []
    if len(os.listdir(path)) == 0:
        print(f"Path {path} is empty")
        return []
    
    images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') and 'only' not in f and 'grid' not in f]
    return images[range_left:range_right]

def calc_non_transparent_pixel_count(image):
    if type(image) == str:
        image = Image.open(image)
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    if type(image) == Image.Image:
        image = image
    assert image.mode == 'RGBA', "Image is not in RGBA mode"
    img_arr = np.array(image)
    ntp = np.where(img_arr[:, :, 3] != 0)
    return len(ntp[0])

def white_to_transparent(image, threshold=250):
    pil_img = Image.fromarray(image)
    assert pil_img.mode == 'RGBA'
    datas = pil_img.getdata()
    new_image_data = []
    for item in datas:
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            new_image_data.append((255, 255, 255, 0))
        else:
            new_image_data.append(item)
    pil_img.putdata(new_image_data)
    return pil_img

def get_results(image, path, model, _class, conf, fdc=True):

    run_path = '/home/floris/Projects/NTNU/models/runs/segment'
    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    if fdc:
        image = find_dominant_color(image)[1]
    results = model(image, retina_masks=True, verbose=False, conf=conf)

    processed_results = []

    for res in results:
        try:
            boxes = res.boxes.xyxy.cpu().numpy()
        except:
            print(f"No boxes found for {os.path.basename(path)}")
            continue
        try:
            masks = res.masks.data.cpu().numpy()
        except:
            print(f"No masks found for {os.path.basename(path)}")
            continue
        
        original = res.orig_img
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        masked_cropped_images = apply_crop_mask(original, masks, boxes)
        image_path = path

        scale_only_path = image_path.replace('.jpg', '_scale_only.jpg')
        if _class == 'fixed':
            px_per_cm = measure_scale_fixed_via_colorboard(scale_only_path)
        if _class == 'random':
            px_per_cm = measure_scale_random(scale_only_path)

        all_boxes = []
        for box in boxes:
            w_cm, h_cm = transform_px_to_cm(box, px_per_cm)
            all_boxes.append((box, {'width_cm': w_cm, 'height_cm': h_cm}))
        
        all_mask_crop_with_sq_cm = []
        for m_img, c_img in masked_cropped_images:
            pil_img = white_to_transparent(c_img)
            ntpx = calc_non_transparent_pixel_count(pil_img)
            area = calculate_mask_area(ntpx, px_per_cm)
            all_mask_crop_with_sq_cm.append((m_img, c_img, area))
        
        processed_results.append({'image': original, 'image_path': image_path, 'px_per_cm': px_per_cm, 'boxes': all_boxes, 'masks_crops_sqcm': all_mask_crop_with_sq_cm})
        
    return processed_results

def save_results(path, save_path, range_left=0, range_right=None):
    check_directory(save_path)

    images = get_images(path)
    print(f"Found {len(images)} image(s) in {path}")
    class_name = path.split('/')[-1]
    PIL_images = [Image.open(i) for i in images]

    for idx, image in enumerate(PIL_images[range_left:range_right]):
        
        results = get_results(image, images[idx], model_seg, class_name, 0.7, fdc=True)

        try:
            res = results[0]
        except IndexError:
            
            try:
                results = get_results(image, images[idx], model_seg, class_name, 0.6, fdc=False)
                res = results[0]
            except IndexError:
                print(f"No results found for {os.path.basename(images[idx])}")
                continue
                
        
        image = Image.fromarray(res["image"])
        image_name = os.path.basename(res["image_path"])

        image_save_path = os.path.join(save_path, image_name.replace('.jpg', '').replace('_a', '').replace('_b', '').replace('_c', ''))
        check_directory(image_save_path)

        image.save(os.path.join(image_save_path, image_name))

        save_dict = {"px_per_cm": res["px_per_cm"], "boxes": [], "sq_cm": []}

        for idx, box in enumerate(res["boxes"]):
            box_sep = box[0].tolist()
            cm_dict = box[1]
            save_dict["boxes"].append({f"box_{idx}": box_sep, "width_cm": cm_dict["width_cm"], "height_cm": cm_dict["height_cm"]})

        for idx2, mask_crop in enumerate(res["masks_crops_sqcm"]):
            crop = white_to_transparent(mask_crop[1])
            crop_name = image_name.replace('.jpg', f'_crop_{idx2}.png')
            crop.save(os.path.join(image_save_path, crop_name))

            area = mask_crop[2]
            save_dict["sq_cm"].append({"id": idx2, "area": area})
        
        json_file_name = image_name.replace('.jpg', '.json')
        with open(os.path.join(image_save_path, json_file_name), 'w') as f:
            json.dump(save_dict, f)
        
        # move cropped scale image to processed folder
        scale_only_path = images[idx].replace('.jpg', '_scale_only.jpg')
        shutil.copy(scale_only_path, image_save_path)

save_results(IMG_PATH_RANDOM, SAVE_PATH_RANDOM, range_left=0, range_right=None)
save_results(IMG_PATH_FIXED, SAVE_PATH_FIXED, range_left=0, range_right=None)