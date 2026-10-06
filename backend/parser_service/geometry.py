import cv2
import numpy as np
from typing import Optional
from backend.schemas import VegNonVegMark

def analyze_veg_mark(image_path: str, bbox: dict) -> Optional[VegNonVegMark]:
    """
    Analyzes the veg/non-veg mark using OpenCV.
    bbox format expected: {'x_min': x, 'y_min': y, 'x_max': a, 'y_max': b}
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    x_min = max(0, bbox.get('x_min', 0))
    y_min = max(0, bbox.get('y_min', 0))
    x_max = bbox.get('x_max', img.shape[1])
    y_max = bbox.get('y_max', img.shape[0])
    
    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
        
    # Convert to HSV to detect color
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Simple color detection (Green vs Brown)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    green_pixels = cv2.countNonZero(green_mask)
    brown_pixels = cv2.countNonZero(brown_mask)
    
    color_detected = None
    if green_pixels > brown_pixels and green_pixels > 50:
        color_detected = "green"
    elif brown_pixels > green_pixels and brown_pixels > 50:
        color_detected = "brown"
        
    # Shape detection (simplified)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_detected = None
    for contour in contours:
        # Approximate contour polygon
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            shape_detected = "triangle"
            break
        elif len(approx) == 4:
            # Distinguish square from rectangle
            _, _, w, h = cv2.boundingRect(approx)
            if 0.8 <= w/float(h) <= 1.2:
                shape_detected = "square"
            break
        elif len(approx) > 4:
            shape_detected = "circle"
            break
            
    # Calibration (mocked for MVP: assume 10 pixels = 1mm for hackathon demo)
    measured_mm = float(x_max - x_min) / 10.0
    
    return VegNonVegMark(
        shape_detected=shape_detected,
        color_detected=color_detected,
        measured_mm=measured_mm,
        required_mm=None,
        pdp_area_cm2=None
    )
