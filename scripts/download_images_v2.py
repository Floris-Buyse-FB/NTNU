import pandas as pd
import requests
import argparse
import random
import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from packages import log, check_directory, get_data, number_to_letter


# CONSTANTS
IMG_DIR = './images'
DATA_DIR = './data'
SAVE_PATH = os.path.join(IMG_DIR, 'gbif_images')
URL_SAVE_PATH = os.path.join(DATA_DIR, 'image_urls.json')
CPU_COUNT = os.cpu_count()
FILENAME = os.path.basename(__file__)


def split_data_get_ids(data: pd.DataFrame, year: int) -> tuple:
    """
    Split the data based on the year and return two lists of gbifIDs.
    The year 2014 is used in this script as it is the year where the random scales were introduced.

    Args:
        data (pd.DataFrame): The input data.
        year (int): The year to split the data.

    Returns:
        tuple: A tuple containing two lists of gbifIDs.
    """
    scale_random = list(data[data['year'] > year]['gbifID'])
    if len(scale_random) == 0:
        # If there are no gbifIDs after the given year, give a warning
        # empty list is returned for scale_fixed
        log(f'Warning: No data after {year} found', FILENAME)
    scale_fixed = list(data[data['year'] <= year]['gbifID'])
    return scale_random, scale_fixed


def get_img_json(ids: list, n_samples=None) -> list:
    """
    Get the JSON data for the given gbifIDs.
    n_samples is used to randomly sample the gbifIDs. 
    If n_samples is None, all the given gbifIDs are used.

    Args:
        ids (list): List of gbifIDs.
        n_samples (int, optional): Number of samples to retrieve. Defaults to None.

    Returns:
        list: List of JSON data for the images.
    """
    if n_samples is not None:
        # Randomly sample the gbifIDs
        n_samples = min(n_samples, len(ids))
        random_list = random.sample(ids, n_samples)
        random_list = sorted(random_list)
        
        # Remove the sampled gbifIDs from the CSV file
        gbif_ids = pd.read_csv(os.path.join(DATA_DIR, 'clean_data.csv'), sep=',')
        gbif_ids = gbif_ids[~gbif_ids['gbifID'].isin(random_list)]
        gbif_ids.to_csv(os.path.join(DATA_DIR, 'clean_data.csv'), sep=',', index=False)
        ids = random_list

    # Get the JSON data for the gbifIDs using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        results = list(executor.map(lambda x: requests.get(f"https://api.gbif.org/v1/occurrence/{x}").json(), ids))
    return results


def get_img_links(img_json: list) -> list:
    """
    Get the image links from the JSON data.

    Args:
        img_json (list): List of JSON data for the images.

    Returns:
        list: List of tuples containing the image links and corresponding names.
    """
    links = []
    # loop over the list of JSON data and get the image links
    for img in img_json:
        list_with_img_links = img['extensions']['http://rs.gbif.org/terms/1.0/Multimedia']
        gbifId = img['gbifID']
        # this loop is to check for multiple images for a single gbifID
        for idx, link in enumerate(list_with_img_links):
            if len(list_with_img_links) > 1:
                # adds a letter to the name if there are multiple images for a single gbifID (a, b, c, ...)
                name = f'{gbifId}_{number_to_letter(idx+1)}'
                links.append((link['http://purl.org/dc/terms/identifier'], name))
            else:
                links.append((link['http://purl.org/dc/terms/identifier'], gbifId))
    return links


def save_urls_to_json(url_id_pairs: list) -> None:
    """
    Save the image URLs to a JSON file.

    Args:
        url_id_pairs (list): List of tuples containing the image URLs and corresponding names.
        
    Returns:
        None
    """
    # Create a dictionary from the list of tuples
    data = {id_: url for url, id_ in url_id_pairs}
    
    # Check if the file exists
    if not os.path.exists(URL_SAVE_PATH):
        # create an empty file
        with open(URL_SAVE_PATH, 'w') as file:
            pass
    
    # Open a file in write mode
    with open(URL_SAVE_PATH, 'a') as file:
        # Write the dictionary to file in JSON format
        json.dump(data, file, indent=4)


def download_img(url_id_pair: tuple, save_path: str) -> None:
    """
    Download the image from the given URL and save it to the specified path.

    Args:
        url_id_pair (tuple): Tuple containing the image URL and corresponding gbifID.
        save_path (str): Path to save the downloaded image.
    
    Returns:
        None
    """
    # unpack the tuple
    img_url, gbif_id = url_id_pair # gbif_id is the name of the image (could be the gbifID or gbifID_a, gbifID_b, ...)
    try:
        # download the image
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        file_path = os.path.join(save_path, f"{gbif_id}.jpg")
        with open(file_path, 'wb') as f:
            f.write(response.content)
    except requests.RequestException as e:
        log(f"Error downloading {img_url}: {e}", FILENAME)
        raise SystemExit


def download_images(url_id_pairs: list, save_path: str) -> None:
    """
    Download the images from the given URLs and save them to the specified path.

    Args:
        url_id_pairs (list): List of tuples containing the image URLs and corresponding gbifIDs.
        save_path (str): Path to save the downloaded images.
        
    Returns:
        None
    """
    # Download the images using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        # uses the download_img function to download the images
        futures = [executor.submit(download_img, pair, save_path) for pair in url_id_pairs]
        for future in as_completed(futures):
            future.result()


def main(ids=None, n_images: int = 33, random: bool = True, save_json: bool = True) -> None:
    """
    Main function to download images from the GBIF API.

    Args:
        ids (list, optional): List of gbifIDs to download images for. Defaults to None.
        n_images (int, optional): Number of images to download for each category. Defaults to 33.
        random (bool, optional): Flag to download images randomly from the CSV file. Defaults to True.
        
    Returns:
        None
    """
    
    # Function to check if the directories exist and create them if they don't
    check_directory(SAVE_PATH)
    check_directory(IMG_DIR)
    check_directory(DATA_DIR)

    # If random is True, download images randomly from the CSV file
    if random:
        try:
            # Retrieve the data from the CSV file
            data = get_data(os.path.join(DATA_DIR, 'clean_data.csv'), ',')
        except Exception as e:
            log(f'Error reading data: {e}', FILENAME)
            raise SystemExit

        try:
            # Split the data based on the year
            scale_random, scale_fixed = split_data_get_ids(data, 2014)
        except Exception as e:
            log(f'Error splitting data: {e}', FILENAME)
            raise SystemExit

        try:
            # Get the JSON data for the gbifIDs
            scale_random_json = get_img_json(scale_random, n_samples=n_images)
            scale_fixed_json = get_img_json(scale_fixed, n_samples=n_images)
        except Exception as e:
            log(f'Error getting image json: {e}', FILENAME)
            raise SystemExit

        try:
            # Get the image links from the JSON data
            scale_random_links = get_img_links(scale_random_json)
            scale_fixed_links = get_img_links(scale_fixed_json)
        except Exception as e:
            log(f'Error getting image links: {e}', FILENAME)
            raise SystemExit
        
        if save_json:
            try:
                # combine the image links
                img_links = scale_random_links + scale_fixed_links
                
                # Save the image URLs to a JSON file
                save_urls_to_json(img_links)
                
            except Exception as e:
                log(f'Error saving URLs to JSON: {e}', FILENAME)
                raise SystemExit

        try:
            # Download the images
            download_images(scale_random_links, SAVE_PATH)
            download_images(scale_fixed_links, SAVE_PATH)
        except Exception as e:
            log(f'Error downloading images: {e}', FILENAME)
            raise SystemExit
    
    # If random is False, download images for the given gbifIDs
    else:
        try:
            img_json = get_img_json(ids)
        except Exception as e:
            log(f'Error getting image json: {e}', FILENAME)
            raise SystemExit

        try:
            img_links = get_img_links(img_json)
        except Exception as e:
            log(f'Error getting image links: {e}', FILENAME)
            raise SystemExit
        
        if save_json:
            try:
                save_urls_to_json(img_links)
            except Exception as e:
                log(f'Error saving URLs to JSON: {e}', FILENAME)
                raise SystemExit

        try:
            download_images(img_links, SAVE_PATH)
        except Exception as e:
            log(f'Error downloading images: {e}', FILENAME)
            raise SystemExit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from GBIF API')
    parser.add_argument('-n', '--n_images', type=int, default=33, help='Number of images to download for each category')
    parser.add_argument('-r', '--random', action='store_true', help='Download images randomly from the CSV file')
    parser.add_argument('-i', '--ids', type=int, nargs='+', help='GBIF IDs to download images for')
    parser.add_argument('-s', '--save_json', action='store_true', help='Save the image URLs to a JSON file')
    args = parser.parse_args()

    if args.random:
        main(n_images=args.n_images, random=True, save_json=args.save_json)
    else:
        if args.ids is None:
            parser.error('Please provide GBIF IDs using the -i or --ids argument when not using random mode')
        main(ids=args.ids, random=False, save_json=args.save_json)