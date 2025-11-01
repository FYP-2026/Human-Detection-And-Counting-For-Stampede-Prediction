
import cv2
import os
import sys
import time

try:
    from src.detection_module import load_model, get_detections
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    sys.exit(1)

MODEL_PATH = os.path.join('trained_models', 'best.pt')
VIDEO_PATH = os.path.join('data', 'test_videos', 'sample.mp4')
CONFIDENCE_THRESHOLD = 0.3

def main():
    
    model = load_model(MODEL_PATH)
    if model is None:
        print("FATAL: Model could not be loaded. Exiting.")
        return

    print(f"Opening video source: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"FATAL: Could not open video file at {VIDEO_PATH}. Exiting.")
        return

    print("INFO: Starting video processing loop (Detection only)...")
    frame_count = 0

    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            print("INFO: End of video file reached")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detected_objects = get_detections(frame, model, confidence_threshold=CONFIDENCE_THRESHOLD)

        debug_frame = frame.copy()
        

        count_text = f"Detected: {len(detected_objects)}"
        cv2.putText(debug_frame, count_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        cv2.putText(debug_frame, f"FPS: {fps:.2f}", (debug_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Detection Output - Press 'q' to quit", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("INFO: Exiting application.")
            break

        frame_count += 1

    # 4. Cleanup
    print("INFO: Releasing video capture and destroying windows.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Basic checks
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        sys.exit(1)
        
    main()