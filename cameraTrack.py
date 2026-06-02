import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicensePlateReader:
    def __init__(self, model_path: str, lang: str = "en"):
        print(f"Loading YOLO model from {model_path}...")
        self.detector = YOLO(model_path)
        print(f"Loading PaddleOCR with language '{lang}'...")
        self.ocr = PaddleOCR(lang=lang)

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the cropped license plate image to improve OCR accuracy.
        """
        return plate_img

    def read_plate(self, image_path: str, output_path: str = "result.jpg"):
        """
        Detect license plates in an image and read their text using OCR.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nProcessing image: {image_path}")
        results = self.detector(image)
        
        all_text = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Ensure coordinates are within image boundaries with padding
                h, w, _ = image.shape
                pad = 10
                x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
                x2_pad, y2_pad = min(w, x2 + pad), min(h, y2 + pad)

                plate = image[y1_pad:y2_pad, x1_pad:x2_pad]
                if plate.size == 0:
                    continue

                # cv2.imwrite(f"plate_{i}.jpg", plate)

                # Preprocess the plate image for better OCR
                processed_plate = self.preprocess_plate(plate)

                # Perform OCR
                # Adding warning suppression or using standard API
                ocr_result = self.ocr.ocr(processed_plate)
                
                plate_text = ""
                if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                    result_item = ocr_result[0]
                    # PaddleX returns a dict with 'rec_texts' key
                    if isinstance(result_item, dict) and 'rec_texts' in result_item:
                        texts = result_item['rec_texts']
                        # Filter out very short or non-alphanumeric noise if needed
                        plate_text = " ".join([t for t in texts if t.strip()])
                    # Fallback for standard PaddleOCR format
                    elif isinstance(result_item, list):
                        for line in result_item:
                            try:
                                text = line[1][0]
                                plate_text += text + " "
                            except Exception:
                                pass

                plate_text = plate_text.strip()
                if plate_text:
                    print(f"Detected Plate [{i}]: {plate_text}")
                    all_text.append(plate_text)

                # Draw bounding box and text on the original image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add background for text to make it readable
                (text_w, text_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(image, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imwrite(output_path, image)
        print(f"Saved results to: {output_path}")
        
        return all_text

    def read_from_camera(self):
        """
        Read from camera directly, detect license plates and display them in real-time.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            results = self.detector(frame, verbose=False)
            plate_detected = False
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    plate_detected = True
                    break

            if not plate_detected:
                cv2.imshow("License Plate Detection (Camera)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Plate detected! Wait for 2 seconds to let camera auto-focus / clear up
            print("Plate detected! Stabilizing camera for 2 seconds...")
            start_time = cv2.getTickCount()
            fps = cv2.getTickFrequency()
            
            # Keep reading and displaying live frames for 2 seconds
            while (cv2.getTickCount() - start_time) / fps < 2.0:
                ret, temp_frame = cap.read()
                if not ret:
                    break
                
                display_frame = temp_frame.copy()
                cv2.putText(display_frame, "Stabilizing... (2s)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("License Plate Detection (Camera)", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # After 2 seconds, grab the final clear frame
            ret, final_frame = cap.read()
            if not ret:
                break
                
            # Pause camera visually & process this final frame
            process_display = final_frame.copy()
            cv2.putText(process_display, "Processing OCR...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("License Plate Detection (Camera)", process_display)
            cv2.waitKey(1) # Force UI update to show paused processing frame
            
            # Run detection on the final clear frame
            results = self.detector(final_frame, verbose=False)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    h, w, _ = final_frame.shape
                    pad = 10
                    x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
                    x2_pad, y2_pad = min(w, x2 + pad), min(h, y2 + pad)

                    plate = final_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    if plate.size == 0:
                        continue

                    # Preprocess and Perform OCR
                    processed_plate = self.preprocess_plate(plate)
                    ocr_result = self.ocr.ocr(processed_plate)
                    
                    plate_text = ""
                    if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                        result_item = ocr_result[0]
                        if isinstance(result_item, dict) and 'rec_texts' in result_item:
                            texts = result_item['rec_texts']
                            plate_text = " ".join([t for t in texts if t.strip()])
                        elif isinstance(result_item, list):
                            for line in result_item:
                                try:
                                    text = line[1][0]
                                    plate_text += text + " "
                                except Exception:
                                    pass

                    plate_text = plate_text.strip()
                    if plate_text:
                        print(f"Detected Plate: {plate_text}")
                        # Draw bounding box and text on the final frame
                        cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (text_w, text_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(final_frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                        cv2.putText(final_frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Show final result and wait a few seconds
            cv2.imshow("License Plate Detection (Camera)", final_frame)
            key = cv2.waitKey(3000) & 0xFF
            
            # Clear buffer before resuming
            for _ in range(30):
                cap.grab()
                
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Using the better named detector model
    MODEL_PATH = "license_plate_detector.pt"
    # MODEL_PATH = "best.pt"
    
    try:
        # Move model loading outside the loop so it only loads once
        reader = LicensePlateReader(model_path=MODEL_PATH)
        
        # Uncomment below lines to run on images
        # IMAGE_PATH = ["test/car1.png","test/car2.png","test/car3.png","test/car4.png","test/car5.png","test/bike1.png"]
        # for img in IMAGE_PATH:
        #     texts = reader.read_plate(image_path=img)
        #     print("\n--- Final Results ---")
        #     for idx, text in enumerate(texts):
        #         print(f"Plate {idx + 1}: {text}")
        
        # Run directly from camera
        print("Starting camera feed...")
        reader.read_from_camera()
            
    except Exception as e:
        print(f"Error occurred: {e}")
