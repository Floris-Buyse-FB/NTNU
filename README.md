# NTNU

## Scripts

``download_images.py``

- Downloads images from the GBIF API
  - Usage: `python download_images.py -n <number of images>` (-n = --n_images)
    - Example: `python download_images.py -n 10`
    - Note: This will download 20 images (10 fixed scale images and 10 random scale images)
    - Default: 33 images
- The images are saved in images/scale_fixed and images/scale_random

``download_single_img.py``

- Downloads a single image from the GBIF API
  - Usage: `python download_single_img.py -i <image id>` (-i = --id)
    - Example: `python download_single_img.py -i 123456789`
- The image is saved in images/test
- This script is more for testing purposes

``classify_images.py``

- Classifies images using a pre-trained model
  - Usage: `python classify_images.py`
- Classified images are saved in images/classification/subfolder
- Subfolders are:
  - fixed
  - random
  - no_class

``crop_fixed_scales.py``