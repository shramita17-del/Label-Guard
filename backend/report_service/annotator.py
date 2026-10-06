import cv2
import base64
import numpy as np
from backend.schemas import ComplianceReport, CheckStatus

def annotate_image(image_path: str, report: ComplianceReport) -> str:
    """
    Draws bounding boxes based on the check results.
    Returns base64 encoded image string.
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create a blank dummy image if file not found (for rapid testing without assets)
        img = np.zeros((800, 800, 3), dtype=np.uint8)

    # Colors (BGR format for OpenCV)
    colors = {
        CheckStatus.PASS: (0, 255, 0),       # Green
        CheckStatus.FAIL: (0, 0, 255),       # Red
        CheckStatus.REVIEW: (0, 165, 255),   # Orange
        CheckStatus.EXEMPT: (255, 255, 255),
        CheckStatus.NOT_APPLICABLE: (200, 200, 200)
    }

    for check in report.check_results:
        bbox = check.evidence_bbox
        if bbox:
            color = colors.get(check.status, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color, 2)
            
            # Add label above bounding box
            label = f"{check.check_name}: {check.status.value}"
            cv2.putText(img, label, (bbox.x_min, max(0, bbox.y_min - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Encode to base64 for embedding directly into HTML
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"
