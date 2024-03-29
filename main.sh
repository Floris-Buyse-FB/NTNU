#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
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

# Check if n_images argument is provided
if [ -z "$1" ]; then
    echo "=============================================================="
    echo "= No argument for n_images provided. Using default value: 20 ="
    echo "=============================================================="
    n_images=20
elif ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "===================================================="
    echo "= Invalid argument for n_images provided. Exiting. ="
    echo "===================================================="
    exit 1
# if the amount of images is greater than 50, ask the user if they want to continue (y/n) and warn them that the system may run out of memory
elif [ "$1" -gt 50 ]; then
    read -p "Warning: The system may run out of memory. Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        n_images=$1
    else
        echo "==========="
        echo "= Exiting ="
        echo "==========="
        exit 1
    fi
else
    n_images=$1
fi

# Ensure scripts are executable
chmod +x ./scripts/download_images.py
chmod +x ./scripts/classify_images.py
chmod +x ./scripts/crop_fixed_scales.py
chmod +x ./scripts/crop_random_scales.py
chmod +x ./scripts/crop_random_v2.py

# Run download command with n_images argument
python3 ./scripts/download_images.py --n_images=$n_images
if [ $? -ne 0 ]; then
    echo "======================================"
    echo "= Error downloading images. Exiting. ="
    echo "======================================"
    exit $?
fi
echo "==================================="
echo "= Downloaded images successfully. ="
echo "==================================="

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

# Initial cropping with confidence = 0.7
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

# Initial cropping with confidence = 0.6
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

# Initial cropping with confidence = 0.5
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

# Deactivate environment and finish
deactivate
echo "==============================="
echo "= Script execution completed. ="
echo "==============================="
