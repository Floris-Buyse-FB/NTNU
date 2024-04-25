import sys
sys.path.append('..')

import numpy as np
import shutil
import torch
import json
import cv2
import os

from ultralytics import YOLO
from packages import check_directory
from sklearn.cluster import KMeans
from PIL import Image

# Clear CUDA cache
torch.cuda.empty_cache()
CUDA = torch.cuda.is_available()
print("CUDA is available:", CUDA)

# CONSTANTS
MODEL_PATH_SEGMENT = './models/plant_segmentation_v20_best.pt'

IMG_PATH_FIXED = './images/cropped_scales/fixed'
IMG_PATH_RANDOM = './images/cropped_scales/random'
SAVE_PATH_RANDOM = './data/processed/random'
SAVE_PATH_FIXED = './data/processed/fixed'
NO_PROCESS_PATH = './data/could_not_process'
URL_PATH = './data/image_urls.json'

DEVICE = "cuda" if CUDA else "cpu"

# Load the segmentation model
model_seg = YOLO(MODEL_PATH_SEGMENT)


def measure_scale_random(image_path: str) -> float:
    """
    Function to measure the scale of an image with a random scale
    
    Args:
        image_path (str): Path to the image to measure the scale of
        
    Returns:
        float: The pixels per cm of the scale in the image
    """
    # Read the image and convert it to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find the contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _max = 0
    _max_idx = 0
    
    # Loop over the contours to find the largest square contour
    for idx, c in enumerate(contours):
        # retrieve the width and height of the bounding rectangle of the contour
        _, _, w, h = cv2.boundingRect(c)
        
        if h > w:
            # check if the width and height are close to each other and the width is at least 18% of the image height
            if abs(w - h) < 10 and w >= 0.18 * image.shape[0]:
                _max_idx = idx if w > _max else _max_idx
                _max = max(w, _max)
        else:
            # check if the width and height are close to each other and the height is at least 18% of the image width
            if abs(w - h) < 10 and w >= 0.18 * image.shape[1]:
                _max_idx = idx if h > _max else _max_idx
                _max = max(h, _max)

    # retrieve the width and height of the bounding rectangle of the largest square contour
    _, _, w, h = cv2.boundingRect(contours[_max_idx])

    # calculate the pixels per cm, where we take the average of the width and height
    px_per_cm_w = w / 2
    px_per_cm_h = h / 2
    px_per_cm = (px_per_cm_w + px_per_cm_h) / 2

    return px_per_cm


def measure_scale_fixed_via_colorboard(image_path: str, box_width_mm: float = 6.2) -> float:
    """
    Function to measure the scale of an image with a fixed scale
    
    Args:
        image_path (str): Path to the image to measure the scale of
        box_width_mm (float): Width of the box in mm (manually measured, the box represents 1 color on the colorboard of the fixed scale images)
    
    Returns:
        float: The pixels per cm of the scale in the image
    """
    # Read the image and convert it to RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds of the red color
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    
    # Create a mask for the red color
    mask = cv2.inRange(image_rgb, lower_red, upper_red)

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Retrieve the width of the bounding rectangle of the largest contour
    _, _, w, _ = cv2.boundingRect(largest_contour)
    
    # Calculate the pixels per cm
    pixels_per_cm = w / (box_width_mm / 10)
    return pixels_per_cm


def transform_px_to_cm(box, px_per_cm: float) -> tuple:
    """
    Function to transform the width and height of a box from pixels to cm
    
    Args:
        box (torch.Tensor): The box to transform
        px_per_cm (float): The pixels per cm of the image
    
    Returns:
        tuple: The width and height of the box in cm
    """
    try:
        # Retrieve the width and height of the box, .cpu() is used to move the tensor to the CPU
        w = np.abs((box[2] - box[0]).cpu())
        h = np.abs((box[3] - box[1]).cpu())
    except:
        # if the tensor is already on the CPU, we can just use the numpy function
        w = np.abs(box[2] - box[0])
        h = np.abs(box[3] - box[1])
    return w / px_per_cm, h / px_per_cm


def get_masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to an image with transparency
    
    Args:
        image (np.ndarray): The image to apply the mask to
        mask (np.ndarray): The mask to apply to the image
    
    Returns:
        np.ndarray: The image with the mask applied
    """
    # Remove single-dimensional entry from the shape of the mask
    mask_squeezed = np.squeeze(mask)  # This should change mask shape from [1, H, W] to [H, W]
    
    # Generate an alpha channel where mask is True (255) and False (0)
    alpha_channel = np.where(mask_squeezed, 255, 0).astype(np.uint8)
    
    # Ensure alpha channel is correctly shaped [H, W] -> [H, W, 1]
    alpha_channel_shaped = np.expand_dims(alpha_channel, axis=-1)

    # Concatenate the alpha channel with the image to create an RGBA image (transparency)
    rgba_image = np.concatenate((image, alpha_channel_shaped), axis=-1)
    return rgba_image


def get_cropped_image(image: np.ndarray, box) -> np.ndarray:
    """
    Crop an image with a given box
    
    Args:
        image (np.ndarray): The image to crop
        box: The box to use for cropping -> x1, y1, x2, y2 format
    
    Returns:
        np.ndarray: The cropped image
    """
    # check if the box is a list or a tensor
    if isinstance(box, list):
        box = box[0]
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy() 
    else:
        box = np.array(box) 

    # Retrieve the x, y, width, and height of the box
    # The box is in the format [x1, y1, x2, y2]
    if len(box.shape) > 1:
        box = box[0]
    x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
    
    # Crop the image with the box coordinates and return the cropped image
    return image[y:y+h, x:x+w]


def apply_crop_mask(image: np.ndarray, mask: np.ndarray, box) -> list:
    """
    Apply a mask to an image and crop the image with a given box
    
    Args:
        image (np.ndarray): The image to apply the mask to
        mask (np.ndarray): The mask to apply to the image
        box: The box to use for cropping -> x1, y1, x2, y2 format
    
    Returns:
        list: A list of tuples containing the masked image and the cropped image
    """
    images = []
    # check if there are multiple masks
    if len(np.array(mask).shape) == 3:
        
        # loop over the masks and boxes
        for i, m in enumerate(mask):
            
            # apply the mask to the image
            m_img = get_masked_image(image, m)
            
            # retrieve the box
            b = box[i].cpu().numpy() if type(box) == torch.Tensor else box[i]
            
            # crop the image with the box
            crop_img = get_cropped_image(m_img, b)
            
            # append the masked image and cropped image as a tuple, to the list
            images.append((m_img, crop_img))
    else:
        # if there is only one mask, apply the mask to the image
        m_img = get_masked_image(image, mask)
        
        # retrieve the box
        b = box.cpu().numpy() if type(box) == torch.Tensor else box
        
        # crop the image with the box
        crop_img = get_cropped_image(m_img, b)
        
        # append the masked image and cropped image as a tuple, to the list
        images.append((m_img, crop_img))
        
    return images

def find_dominant_color(image: Image, k: int = 5) -> tuple:
    """
    Find the dominant color in an image and turn the dominant color range to white
    
    Args:
        image (Image): The image to find the dominant color in (PIL Image)
        k (int): The number of clusters to use for k-means
    
    Returns:
        tuple: The dominant color and the image with the dominant color range turned to white (PIL Image)
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Reshape it to a list of RGB values
    img_vector = img_array.reshape((-1, 3))
    
    # Run k-means on the pixel colors (fit only on a subsample to speed up)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_vector[::50])
    
    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    # Create a mask for pixels within a certain distance from the dominant color
    # uses the standard deviation of the distances as the threshold
    distances = np.sqrt(np.sum((img_vector - dominant_color) ** 2, axis=1))
    mask = distances < np.std(distances)
    
    # Turn the dominant color range to white
    img_vector[mask] = [255, 255, 255]
    result_img_array = img_vector.reshape(img_array.shape)
    
    # Turn image back to PIL
    result_img = Image.fromarray(result_img_array.astype(np.uint8))
    return dominant_color, result_img


def calculate_mask_area(masked_pixels: int, pixels_per_cm: float) -> float:
    """
    Calculate the area of a mask in square cm
    
    Args:
        masked_pixels (int): The number of masked pixels
        pixels_per_cm (float): The number of pixels per cm
        
    Returns:
        float: The area of the mask in square cm
    """
    area_square_cm = masked_pixels / (pixels_per_cm ** 2)
    return area_square_cm


def get_images(path: str, range_left: int = 0, range_right: int = None) -> list:
    """
    Get all images in a directory
    
    Args:
        path (str): The path to the directory
        range_left (int): The left range of the images to get
        range_right (int): The right range of the images to get
        
    Returns:
        list: A list of image paths
    """
    # Check if the path exists
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        return []

    # Check if the path is empty
    if len(os.listdir(path)) == 0:
        print(f"Path {path} is empty")
        return []
    
    # Get all images in the directory that end with .jpg and do not contain 'only' or 'grid' in the name
    images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') and 'only' not in f and 'grid' not in f]
    
    # Return the images in the specified range
    return images[range_left:range_right]


def calc_non_transparent_pixel_count(image) -> int:
    """
    Calculate the number of non-transparent pixels in an RGBA image
    
    Args:
        image: The image to calculate the number of non-transparent pixels in
    
    Returns:
        int: The number of non-transparent pixels
    """
    # Check if the image is a string (path), numpy array, or PIL Image
    # and convert it to a PIL Image if it is not already
    if type(image) == str:
        image = Image.open(image)
        
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
        
    if type(image) == Image.Image:
        image = image
    
    # Check if the image is in RGBA mode (transparency)
    assert image.mode == 'RGBA', "Image is not in RGBA mode"
    
    # Convert the image to a numpy array
    img_arr = np.array(image)
    
    # Find the non-transparent pixels
    ntp = np.where(img_arr[:, :, 3] != 0)
    return len(ntp[0])


def white_to_transparent(image: np.ndarray, threshold: int = 250) -> Image:
    """
    Convert white pixels in an image to transparent
    
    Args:
        image (np.ndarray): The image to convert
        threshold (int): The threshold to use for converting white pixels to transparent
        
    Returns:
        Image: The image with white pixels converted to transparent (PIL Image)
    """
    # Convert the image to a PIL Image
    pil_img = Image.fromarray(image)
    
    # Check if the image is in RGBA mode (transparency)
    assert pil_img.mode == 'RGBA'
    
    # Get the image data
    datas = pil_img.getdata()
    
    new_image_data = []
    # Loop over the image data and convert white pixels to transparent
    for item in datas:
        
        # Check if the pixel is white (greater than the threshold for all RGB values)
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            new_image_data.append((255, 255, 255, 0))
        else:
            # If the pixel is not white, keep the pixel as is
            new_image_data.append(item)
        
    # Put the new image data back into the image
    pil_img.putdata(new_image_data)
    
    # Return the image with white pixels converted to transparent as a PIL Image
    return pil_img


def get_results(image: Image, path: str, model: YOLO, _class: str, conf: float, fdc: bool = True) -> list:
    """
    Get the results of the segmentation model and process the results
    This is only used for 1 image at a time
    
    Args:
        image (Image): The image to get the results of
        path (str): The path to the image
        model (YOLO): The segmentation model
        _class (str): The class of the image (should be 'fixed' or 'random')
        conf (float): The confidence threshold to use
        fdc (bool): Whether to use find_dominant_color or not
    
    Returns:
        list: The processed results
    """
    
    # Check if the class is 'fixed' or 'random'
    assert _class in ['fixed', 'random'], "Class should be 'fixed' or 'random'"
    
    # Check if the path exists and remove the run path if it exists
    run_path = '../models/runs/segment'
    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    # If fdc is True, find the dominant color of the image and turn the dominant color range to white
    if fdc:
        image = find_dominant_color(image)[1]
    
    # Get the results of the model, retina_masks=True is used to get high quality masks
    results = model(image, retina_masks=True, verbose=False, conf=conf)

    processed_results = []

    # Loop over the results
    for res in results:
        
        # Check if the boxes and masks are found
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
        
        # Get the original image
        original = res.orig_img
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Get the masked and cropped images
        masked_cropped_images = apply_crop_mask(original, masks, boxes)
        
        # Get the path of the image
        image_path = path

        # Measure the scale of the image depending on the class
        scale_only_path = image_path.replace('.jpg', '_scale_only.jpg')
        if _class == 'fixed':
            px_per_cm = measure_scale_fixed_via_colorboard(scale_only_path)
        if _class == 'random':
            px_per_cm = measure_scale_random(scale_only_path)

        # Loop over the boxes and transform the width and height to cm
        all_boxes = []
        for box in boxes:
            w_cm, h_cm = transform_px_to_cm(box, px_per_cm)
            all_boxes.append((box, {'width_cm': w_cm, 'height_cm': h_cm}))
        
        # Loop over the tuples of masked and cropped images and calculate the area in square cm
        all_mask_crop_with_sq_cm = []
        for m_img, c_img in masked_cropped_images:
            
            # turn white pixels to transparent
            pil_img = white_to_transparent(c_img)
            
            # calculate the number of non-transparent pixels
            ntpx = calc_non_transparent_pixel_count(pil_img)
            
            # calculate the area in square cm
            area = calculate_mask_area(ntpx, px_per_cm)
            
            # append the masked image, cropped image, and area to the list
            all_mask_crop_with_sq_cm.append((m_img, c_img, area))
        
        # Append the processed results to the list in dictionary format
        processed_results.append({'image': original, 'image_path': image_path, 'px_per_cm': px_per_cm, 'boxes': all_boxes, 'masks_crops_sqcm': all_mask_crop_with_sq_cm})
        
    return processed_results


def save_results(path: str, save_path: str, range_left: int = 0, range_right: int = None) -> None:
    """
    Save the results of the segmentation model to a directory
    
    Args:
        path (str): The path to the directory with images
        save_path (str): The path to save the results to
        range_left (int): The left range of the images to process
        range_right (int): The right range of the images to process
    
    Returns:
        None
    """
    
    # Check if the save path exists and create it if it does not
    check_directory(save_path)

    # Get a list of images in the directory (full path)
    images = get_images(path)
    print(f"Found {len(images)} image(s) in {path}")
    
    # Define the class_name
    class_name = path.split('/')[-1]
    
    # Convert the images to PIL Images
    PIL_images = [Image.open(i) for i in images]

    # Loop over the predefined range of PIL images
    for idx, image in enumerate(PIL_images[range_left:range_right]):
        
        # Get the results of the segmentation model
        results = get_results(image, images[idx], model_seg, class_name, 0.7, fdc=True)

        # Check if results are found and retry with lower confidence thresholds if not
        try:
            res = results[0]
        except IndexError:
            
            try:
                print("Trying to get results with a lower confidence threshold (0.6)")
                results = get_results(image, images[idx], model_seg, class_name, 0.6, fdc=True)
                res = results[0]
            except IndexError:
                
                try:
                    print("Trying to get results with a lower confidence threshold (0.5)")
                    results = get_results(image, images[idx], model_seg, class_name, 0.5, fdc=True)
                    res = results[0]
                except IndexError:
                    
                    try:
                        print("Trying to get results with a lower confidence threshold (0.4)")
                        results = get_results(image, images[idx], model_seg, class_name, 0.4, fdc=True)
                        res = results[0]
                    except IndexError:
                        
                        try:
                            print("Trying to get results with a lower confidence threshold (0.3)")
                            results = get_results(image, images[idx], model_seg, class_name, 0.3, fdc=True)
                            res = results[0]
                        except IndexError:
                                
                                try:
                                    print("Trying to get results with a lower confidence threshold (0.2)")
                                    results = get_results(image, images[idx], model_seg, class_name, 0.2, fdc=True)
                                    res = results[0]
                                except IndexError:
                                    
                                    try:
                                        print("Trying to get results with a lower confidence threshold (0.1)")
                                        results = get_results(image, images[idx], model_seg, class_name, 0.1, fdc=True)
                                        res = results[0]
                                    except IndexError:
                                        print(f"No results found for {os.path.basename(images[idx])}")
                                        
                                        # Move the image to a directory 'could_not_process' if no results are found
                                        check_directory(NO_PROCESS_PATH)
                                        shutil.move(images[idx], NO_PROCESS_PATH)
                                        continue
        
        # open the image from the results with PIL
        image = Image.fromarray(res["image"])
        
        # Define the image name and save path
        image_name = os.path.basename(res["image_path"])
        image_save_path = os.path.join(save_path, image_name.replace('.jpg', '').replace('_a', '').replace('_b', '').replace('_c', ''))
        
        # Check if the save path exists and create it if it does not
        check_directory(image_save_path)
        
        # Save the image
        image.save(os.path.join(image_save_path, image_name))

        # Open the image_urls.json file and load the corresponding image URL
        with open(URL_PATH, 'r') as f:
            urls = json.load(f)
        
        # Define the dictionary to save as a JSON file
        save_dict = {"px_per_cm": res["px_per_cm"], "boxes": [], "sq_cm": [], "img_url": urls[image_name[:-4]]}

        # Loop over the boxes and save the width and height in cm
        for idx, box in enumerate(res["boxes"]):
            
            # every box is a tuple with the box coordinates and a dictionary with the width and height in cm
            box_sep = box[0].tolist()
            cm_dict = box[1]
            
            save_dict["boxes"].append({f"box_{idx}": box_sep, "width_cm": cm_dict["width_cm"], "height_cm": cm_dict["height_cm"]})

        # Loop over the masked and cropped images and save the area in square cm
        for idx2, mask_crop in enumerate(res["masks_crops_sqcm"]):
            
            # mask_crop is a tuple with the masked image (0), cropped image (1), and the area in square cm (2)
            crop = white_to_transparent(mask_crop[1])
            crop_name = image_name.replace('.jpg', f'_crop_{idx2}.png')
            crop.save(os.path.join(image_save_path, crop_name))
            
            area = mask_crop[2]
            
            save_dict["sq_cm"].append({"id": idx2, "area": area})
        
        # Define the JSON file name and save the dictionary as a JSON file
        json_file_name = image_name.replace('.jpg', '.json')
        with open(os.path.join(image_save_path, json_file_name), 'w') as f:
            json.dump(save_dict, f)


def move_and_remove_old_dirs(path: str, save_path: str) -> None:
    """
    Move the processed images to a new directory and remove the old directories
    
    Args:
        path (str): The path to the directory with images
        save_path (str): The path to save the processed images to
        
    Returns:
        None
    """
    
    # Get all images in the directory (full path)
    images = get_images(path)
    
    # Loop over the images
    for image_path in images:
        
        # Define the save directory path
        save_dir_path = os.path.join(save_path, os.path.basename(image_path).replace('.jpg', '').replace('_a', '').replace('_b', '').replace('_c', ''))
        
        # move cropped scale image to processed folder
        scale_only_path = image_path.replace('.jpg', '_scale_only.jpg')
        shutil.move(scale_only_path, save_dir_path)
        
        # move original image to processed folder with _original suffix
        new_image_name_path = os.path.basename(image_path).replace('.jpg', '_original.jpg')
        os.rename(image_path, new_image_name_path)
        shutil.move(new_image_name_path, save_dir_path)
            
    # remove the cropped scales folder
    shutil.rmtree(path)


if __name__ == '__main__':
    save_results(IMG_PATH_RANDOM, SAVE_PATH_RANDOM, range_left=0, range_right=None)
    move_and_remove_old_dirs(IMG_PATH_RANDOM, SAVE_PATH_RANDOM)
    save_results(IMG_PATH_FIXED, SAVE_PATH_FIXED, range_left=0, range_right=None)
    move_and_remove_old_dirs(IMG_PATH_FIXED, SAVE_PATH_FIXED)
    
    # remove the images folder
    shutil.rmtree('../images')
    
    # remove the urls file
    os.remove(URL_PATH)
