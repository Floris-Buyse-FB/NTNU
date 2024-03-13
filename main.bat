@echo off
REM Activate virtual environment
call .venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo ====================================
    echo = Error activating virtual env. Exiting. =
    echo ====================================
    exit /b %ERRORLEVEL%
)

REM Run download command
python ./scripts/download_images.py --n_images=20
IF %ERRORLEVEL% NEQ 0 (
    echo ================================
    echo = Error downloading images. Exiting. =
    echo ================================
    exit /b %ERRORLEVEL%
)
echo ===================================
echo = Downloaded images successfully. =
echo ===================================

REM Classify
python ./scripts/classify_images.py
IF %ERRORLEVEL% NEQ 0 (
    echo ==============================
    echo = Error classifying images. Exiting. =
    echo ==============================
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
echo ============================================
echo = Cropped fixed scales images successfully. =
echo ============================================

python ./scripts/crop_random_scales.py
IF %ERRORLEVEL% NEQ 0 (
    echo ==========================================
    echo = Error cropping random scales. Exiting. =
    echo ==========================================
    exit /b %ERRORLEVEL%
)
echo ============================================
echo = Cropped random scales images successfully.=
echo ============================================

REM Deactivate environment
call .venv\Scripts\deactivate.bat

echo ==============================
echo = Script execution completed. =
echo ==============================

