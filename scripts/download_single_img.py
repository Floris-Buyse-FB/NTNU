from packages import log, check_directory, number_to_letter
import requests
import argparse
import os

# CONSTANTS
IMG_DIR = './images'
DATA_DIR = './data'
SCRIPT_DIR = './scripts'
MODEL_DIR = './models'

CPU_COUNT = os.cpu_count()

SAVE_PATH = os.path.join(IMG_DIR, 'test')


def get_img_json(id):
    result = requests.get(f"https://api.gbif.org/v1/occurrence/{id}").json()
    return result


def get_img_links(img_json):
    "Get image links from the json data."
    links = []
    list_with_img_links = img_json['extensions']['http://rs.gbif.org/terms/1.0/Multimedia']
    gbifId = img_json['gbifID']
    for idx, link in enumerate(list_with_img_links):
        if len(list_with_img_links) > 1:
            name = f'{gbifId}_{number_to_letter(idx+1)}'
            links.append((link['http://purl.org/dc/terms/identifier'], name))
        else:
            links.append((link['http://purl.org/dc/terms/identifier'], gbifId))
    return links


def download_img(url_id_pair, save_path):
    "Download an image from a url and save it to a directory. This is a helper function for download_images."
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


def main(id):
    check_directory(SAVE_PATH)
    check_directory(IMG_DIR)
    check_directory(DATA_DIR)

    try:
        img_json = get_img_json(id)
    except Exception as e:
        log(f'Error getting image json: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit 
    
    try:
        img_links = get_img_links(img_json)
    except Exception as e:
        log(f'Error getting image links: {e}')
        print('Something went wrong, check the log file for more information')
        print(e)
        raise SystemExit 

    try:
        for link in img_links:
            download_img(link, SAVE_PATH)
    except Exception as e:
        log(f'Error downloading image: {e}')
        print('Something went wrong, check the log file for more information')
        raise SystemExit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download images from GBIF")
    parser.add_argument('-i', '--id', type=int, help="The GBIF ID of the species to download images for.")
    args = parser.parse_args()
    main(args.id)