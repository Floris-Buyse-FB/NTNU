from concurrent.futures import ThreadPoolExecutor, as_completed
from packages import log
import pandas as pd
import requests
import random
import json
import os

# Constants
IMG_DIR = './images'
DATA_DIR = './data'
SCRIPT_DIR = './scripts'
MODEL_DIR = './models'

CPU_COUNT = os.cpu_count()

SCALE_FIXED_PATH = os.path.join(IMG_DIR, 'scale_fixed')
SCALE_RANDOM_PATH = os.path.join(IMG_DIR, 'scale_random')


def check_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_data(url, sep):
    try:
        data = pd.read_csv(url, sep=sep)
        return data
    except Exception as e:
        log(f'Error reading data: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit


def split_data_get_ids(data, year):
    scale_random = list(data[data['year'] > year]['gbifID'])
    scale_fixed = list(data[data['year'] <= year]['gbifID'])
    return scale_random, scale_fixed


def get_img_json(ids, n_samples=100):

    n_samples = min(n_samples, len(ids))
    random_list = random.sample(ids, n_samples)
    random_list = sorted(random_list)

    gbif_ids = pd.read_csv(os.path.join(DATA_DIR, 'clean_data.csv'), sep=',')
    gbif_ids = gbif_ids[~gbif_ids['gbifID'].isin(random_list)]
    gbif_ids.to_csv(os.path.join(DATA_DIR, 'clean_data.csv'), sep=',', index=False)
    
    
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        results = list(executor.map(lambda x: requests.get(f"https://api.gbif.org/v1/occurrence/{x}").json(), random_list))
    return results


def get_img_links(img_json):
    links = []
    for img in img_json:
        list_with_img_links = img['extensions']['http://rs.gbif.org/terms/1.0/Multimedia']
        gbifId = img['gbifID']
        for link in list_with_img_links:
            links.append((link['http://purl.org/dc/terms/identifier'], gbifId))
    return links


def download_img(url_id_pair, save_path):
    img_url, gbif_id = url_id_pair
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        
        file_path = os.path.join(save_path, f"{gbif_id}.jpg")
        with open(file_path, 'wb') as f:
            f.write(response.content)
    except requests.RequestException as e:
        log(f"Error downloading {img_url}: {e}")
        print('Something went wrong, check the log file for more information')
        raise SystemExit 


def download_images(url_id_pairs, save_path):
    # Using ThreadPoolExecutor to parallelize downloads
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        # Create a future for each download task
        futures = [executor.submit(download_img, pair, save_path) for pair in url_id_pairs]
        
        # As each future completes, we could check its status or result here
        for future in as_completed(futures):
            future.result()  # This will re-raise any exception caught by the `download_img` function


def main(n_images=20):
    check_directory(SCALE_FIXED_PATH)
    check_directory(SCALE_RANDOM_PATH)
    check_directory(IMG_DIR)
    check_directory(DATA_DIR)

    try:
        data = get_data(os.path.join(DATA_DIR, 'clean_data.csv'), ',')
    except Exception as e:
        log(f'Error reading data: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit 

    try:    
        scale_random, scale_fixed = split_data_get_ids(data, 2014)
    except Exception as e:
        log(f'Error splitting data: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit 
    
    try:
        scale_random_json = get_img_json(scale_random, n_samples=n_images)
        scale_fixed_json = get_img_json(scale_fixed, n_samples=n_images)
    except Exception as e:
        log(f'Error getting image json: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit 

    try:

        scale_random_links = get_img_links(scale_random_json)
        scale_fixed_links = get_img_links(scale_fixed_json)
    except Exception as e:
        log(f'Error getting image links: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit 
    
    try:
        download_images(scale_random_links, SCALE_RANDOM_PATH)
        download_images(scale_fixed_links, SCALE_FIXED_PATH)
    except Exception as e:
        log(f'Error downloading images: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit 


if __name__ == '__main__':
    main()