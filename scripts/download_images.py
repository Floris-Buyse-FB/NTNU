from concurrent.futures import ThreadPoolExecutor, as_completed
from packages import log, check_directory, get_data, number_to_letter
import pandas as pd
import requests
import argparse
import random
import os

# CONSTANTS
IMG_DIR = './images'
DATA_DIR = './data'
SAVE_PATH = os.path.join(IMG_DIR, 'gbif_images')
CPU_COUNT = os.cpu_count()
FILENAME = os.path.basename(__file__)


def split_data_get_ids(data, year):
    scale_random = list(data[data['year'] > year]['gbifID'])
    if len(scale_random) == 0:
        log(f'Warning: No data after {year} found', FILENAME)
    scale_fixed = list(data[data['year'] <= year]['gbifID'])
    return scale_random, scale_fixed


def get_img_json(ids, n_samples):
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
        for idx, link in enumerate(list_with_img_links):
            if len(list_with_img_links) > 1:
                name = f'{gbifId}_{number_to_letter(idx+1)}'
                links.append((link['http://purl.org/dc/terms/identifier'], name))
            else:
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
        log(f"Error downloading {img_url}: {e}", FILENAME)
        raise SystemExit 


def download_images(url_id_pairs, save_path):

    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [executor.submit(download_img, pair, save_path) for pair in url_id_pairs]
        for future in as_completed(futures):
            future.result()


def main(n_images=33):
    check_directory(SAVE_PATH)
    check_directory(IMG_DIR)
    check_directory(DATA_DIR)

    try:
        data = get_data(os.path.join(DATA_DIR, 'clean_data.csv'), ',')
    except Exception as e:
        log(f'Error reading data: {e}', FILENAME)
        raise SystemExit 

    try:    
        scale_random, scale_fixed = split_data_get_ids(data, 2014)
    except Exception as e:
        log(f'Error splitting data: {e}', FILENAME)
        raise SystemExit 
    
    try:
        scale_random_json = get_img_json(scale_random, n_samples=n_images)
        scale_fixed_json = get_img_json(scale_fixed, n_samples=n_images)
    except Exception as e:
        log(f'Error getting image json: {e}', FILENAME)
        raise SystemExit 

    try:

        scale_random_links = get_img_links(scale_random_json)
        scale_fixed_links = get_img_links(scale_fixed_json)
    except Exception as e:
        log(f'Error getting image links: {e}', FILENAME)
        raise SystemExit 
    
    try:
        download_images(scale_random_links, SAVE_PATH)
        download_images(scale_fixed_links, SAVE_PATH)
    except Exception as e:
        log(f'Error downloading images: {e}', FILENAME)
        raise SystemExit 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from GBIF API')
    parser.add_argument('-n', '--n_images', type=int, default=33, help='Number of images to download for each category')
    args = parser.parse_args()
    main(args.n_images)