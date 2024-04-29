#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/packages"

# Export PYTHONPATH
export PYTHONPATH="$PACKAGE_DIR:$PYTHONPATH"

# Activate virtual environment
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "=========================================="
    echo "= Error activating virtual env. Exiting. ="
    echo "=========================================="
    exit $?
fi

# Ensure scripts are executable
chmod +x ./scripts/*.py

# Present options for downloading images
echo "Select the download mode:"
echo "1) Download random images"
echo "2) Download images by specifying GBIF IDs"
read -p "Enter your choice (1 or 2): " download_choice
if [ "$download_choice" == "1" ]; then
    read -p "Enter the number of images to download (default 20, max 50): " n_images
    n_images=${n_images:-20} # Default to 20 if no input
    if ! [[ "$n_images" =~ ^[0-9]+$ ]]; then
        echo "Invalid number provided. Exiting."
        exit 1
    elif [ "$n_images" -gt 50 ]; then
        read -p "Warning: Requesting more than 50 images may cause memory issues. Continue? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting."
            exit 1
        fi
    fi

    # Run download command with n_images argument
    python3 ./scripts/download_images_v2.py -r -s -n "$n_images"
elif [ "$download_choice" == "2" ]; then
    read -p "Enter GBIF IDs separated by space: " -a gbif_ids
    if [ ${#gbif_ids[@]} -eq 0 ]; then
        echo "No GBIF IDs entered. Exiting."
        exit 1
    fi
    python3 ./scripts/download_images_v2.py -s -i "${gbif_ids[@]}"
else
    echo "Invalid choice. Exiting."
    exit 1
fi
if [ $? -ne 0 ]; then
    echo "========================================"
    echo "= Error in download process. Exiting. ="
    echo "========================================"
    exit $?
fi
echo "=================================="
echo "= Downloaded images successfully.="
echo "=================================="

# Classify
python3 ./scripts/classify_images.py
if [ $? -ne 0 ]; then
    echo "====================================="
    echo "= Error classifying images. Exiting.="
    echo "====================================="
    exit $?
fi
echo "====================================="
echo "= Classified images successfully.   ="
echo "====================================="

# Crop images
python3 ./scripts/crop_fixed_scales.py
if [ $? -ne 0 ]; then
    echo "========================================="
    echo "= Error cropping fixed scales. Exiting. ="
    echo "========================================="
    exit $?
fi
echo "============================================="
echo "= Cropped fixed scales images successfully. ="
echo "============================================="

python3 ./scripts/crop_random_scales.py
if [ $? -ne 0 ]; then
    echo "=========================================="
    echo "= Error cropping random scales. Exiting. ="
    echo "=========================================="
    exit $?
fi
echo "================================================"
echo "= Cropped random scales images successfully.  ="
echo "================================================"

# Crop fixed images v2 with confidence = 0.7
python3 ./scripts/crop_fixed_v2.py -c=0.7
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "= Error cropping fixed undetected scales. Exiting. ="
    echo "====================================================="
    exit $?
fi
echo "=================================================="
echo "= Cropped fixed undetected scales successfully. ="
echo "=================================================="

# Crop fixed images v2 with confidence = 0.6
python3 ./scripts/crop_fixed_v2.py -c=0.6
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "= Error cropping fixed undetected scales. Exiting. ="
    echo "====================================================="
    exit $?
fi
echo "=================================================="
echo "= Cropped fixed undetected scales successfully. ="
echo "=================================================="

# Crop fixed images v2 with confidence = 0.5
python3 ./scripts/crop_fixed_v2.py -c=0.5
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "= Error cropping fixed undetected scales. Exiting. ="
    echo "====================================================="
    exit $?
fi
echo "=================================================="
echo "= Cropped fixed undetected scales successfully. ="
echo "=================================================="

# Crop random images v2 with confidence = 0.7
python3 ./scripts/crop_random_v2.py -c=0.7
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "= Error cropping random undetected scales. Exiting. ="
    echo "====================================================="
    exit $?
fi
echo "=================================================="
echo "= Cropped random undetected scales successfully. ="
echo "=================================================="

# Crop random images v2 with confidence = 0.6
python3 ./scripts/crop_random_v2.py -c=0.6
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "= Error cropping random undetected scales. Exiting. ="
    echo "====================================================="
    exit $?
fi
echo "=================================================="
echo "= Cropped random undetected scales successfully. ="
echo "=================================================="

# Crop random images v2 with confidence = 0.5
python3 ./scripts/crop_random_v2.py -c=0.5
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "= Error cropping random undetected scales. Exiting. ="
    echo "====================================================="
    exit $?
fi
echo "=================================================="
echo "= Cropped random undetected scales successfully. ="
echo "=================================================="

# Execute the segmentation script
python3 ./scripts/segment_v2.py
if [ $? -ne 0 ]; then
    echo "====================================="
    echo "= Error segmenting images. Exiting. ="
    echo "====================================="
    exit $?
fi
echo "====================================="
echo "= Segmented images successfully.    ="
echo "====================================="

# Deactivate environment and finish
deactivate
echo "==============================="
echo "= Script execution completed. ="
echo "==============================="
