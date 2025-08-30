@echo off
set INPUT=%1
set OUTPUT=%2
if "%INPUT%"=="" echo Usage: face_strict_win.bat INPUT_DIR OUTPUT_DIR & exit /b 1
if "%OUTPUT%"=="" echo Usage: face_strict_win.bat INPUT_DIR OUTPUT_DIR & exit /b 1
python -m face.cli --strict --input "%INPUT%" --output "%OUTPUT%"
