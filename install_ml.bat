@echo off
echo ===================================================
echo   LabelGuard AI - Machine Learning Setup Script
echo ===================================================
echo.
echo This script will install the heavy ML dependencies required 
echo for Real-Time YOLOv8 Object Detection and PaddleOCR Text Extraction.
echo.
echo Installing Ultralytics (YOLOv8)...
pip install ultralytics

echo.
echo Installing PaddleOCR and its dependencies...
pip install paddlepaddle paddleocr

echo.
echo Downloading YOLOv8 Nano weights (yolov8n.pt)...
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Weights downloaded successfully!')"

echo.
echo ===================================================
echo Setup Complete! 
echo The FastAPI server will now use live models instead of mocks.
echo You can run the server using: uvicorn backend.main:app --reload
echo ===================================================
pause
