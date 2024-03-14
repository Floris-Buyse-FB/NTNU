@echo off
REM Activate virtual environment
call .venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo ==========================================
    echo = Error activating virtual env. Exiting. =
    echo ==========================================
    exit /b %ERRORLEVEL%
)

REM Check if n_images argument is provided
IF "%1"=="" (
    echo ==============================================================
    echo = No argument for n_images provided. Using default value: 20 =
    echo ==============================================================
    SET n_images=20
) ELSE (
    SET n_images=%1
)

REM Run download command with n_images argument
python ./scripts/download_images.py --n_images=%n_images%
IF %ERRORLEVEL% NEQ 0 (
    echo ======================================
    echo = Error downloading images. Exiting. =
    echo ======================================
    exit /b %ERRORLEVEL%
)
echo ===================================
echo = Downloaded images successfully. =
echo ===================================

REM Classify
python ./scripts/classify_images.py
IF %ERRORLEVEL% NEQ 0 (
    echo ======================================
    echo = Error classifying images. Exiting. =
    echo ======================================
    exit /b %ERRORLEVEL%
)
echo =====================================
echo = Classified images successfully.   =
echo =====================================

REM Crop images
python ./scripts/crop_fixed_scales.py
IF %ERRORLEVEL% NEQ 0 (
    echo =========================================
    echo = Error cropping fixed scales. Exiting. =
    echo =========================================
    exit /b %ERRORLEVEL%
)
echo =============================================
echo = Cropped fixed scales images successfully. =
echo =============================================

python ./scripts/crop_random_scales.py
IF %ERRORLEVEL% NEQ 0 (
    echo ==========================================
    echo = Error cropping random scales. Exiting. =
    echo ==========================================
    exit /b %ERRORLEVEL%
)
echo ==============================================
echo = Cropped random scales images successfully. =
echo ==============================================

REM Initial cropping with confidence = 0.7
python ./scripts/crop_random_v2.py -c=0.7
IF %ERRORLEVEL% NEQ 0 (
    echo =====================================================
    echo = Error cropping random undetected scales. Exiting. =
    echo =====================================================
    exit /b %ERRORLEVEL%
)
echo ==================================================
echo = Cropped random undetected scales successfully. =
echo ==================================================

REM Initial cropping with confidence = 0.6
python ./scripts/crop_random_v2.py -c=0.6
IF %ERRORLEVEL% NEQ 0 (
    echo =====================================================
    echo = Error cropping random undetected scales. Exiting. =
    echo =====================================================
    exit /b %ERRORLEVEL%
)
echo ==================================================
echo = Cropped random undetected scales successfully. =
echo ==================================================

REM Initial cropping with confidence = 0.5
python ./scripts/crop_random_v2.py -c=0.5
IF %ERRORLEVEL% NEQ 0 (
    echo =====================================================
    echo = Error cropping random undetected scales. Exiting. =
    echo =====================================================
    exit /b %ERRORLEVEL%
)
echo ==================================================
echo = Cropped random undetected scales successfully. =
echo ==================================================

REM Deactivate environment and finish
call .venv\Scripts\deactivate.bat
echo ===============================
echo = Script execution completed. =
echo ===============================

