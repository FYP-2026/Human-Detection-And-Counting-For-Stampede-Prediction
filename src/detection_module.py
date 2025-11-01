

import os
import sys
from ultralytics import YOLO

def load_model(model_path):
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_detections(frame, model, confidence_threshold):

    bboxes = []
    try:
        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf.item()
                if conf >= confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().tolist() 
                    cls_id = int(box.cls.item())
                    
                    bboxes.append(xyxy + [conf, cls_id])

    except Exception as e:
        print(f"Error during detection: {e}")

    return bboxes

if __name__ == '__main__':
    print("Testing detection_module...")
   