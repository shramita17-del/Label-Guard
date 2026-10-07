@echo off
echo ===================================================
echo   LabelGuard AI - Machine Learning Setup Script
echo ===================================================
echo.
echo This script will install the ML dependencies required 
echo for YOLOv8 Object Detection and PaddleOCR Text Extraction.
echo.

echo [1/3] Installing Ultralytics (YOLOv8)...
pip install ultralytics

echo.
echo [2/3] Downloading YOLOv8 Nano weights (yolov8n.pt)...
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8 weights ready!')"

echo.
echo [3/3] Attempting PaddleOCR installation...
echo Note: On Windows + Python 3.13, pre-built PaddlePaddle wheels may not be available on PyPI yet.
pip install paddlepaddle paddleocr || echo (PaddleOCR optional install skipped or needs Python 3.10-3.12 wheel)

echo.
echo ===================================================
echo Setup Complete!
echo Start the server with: uvicorn backend.main:app
echo ===================================================
pause
