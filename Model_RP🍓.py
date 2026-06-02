import os
# Prevent Segmentation Faults in PaddleOCR on ARM CPUs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

class LicensePlateReader:
    def __init__(self, model_path: str, lang: str = "en"):
        print(f"Loading YOLO model from {model_path}...")
        self.detector = YOLO(model_path)
        print(f"Loading EasyOCR with language '{lang}'...")
        # Use EasyOCR to avoid ARM Segmentation Faults
        self.ocr = easyocr.Reader([lang], gpu=False)

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the cropped license plate image to improve OCR accuracy.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Upscale the image for better OCR reading on small plates
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(resized)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        blur = cv2.bilateralFilter(contrast, 11, 17, 17)
        
        # Ensure the array is contiguous in memory.
        return np.ascontiguousarray(blur)

    def correct_indian_plate(self, text: str) -> str:
        """
        Fixes common OCR confusions (e.g. Z->2, G->6) by enforcing the 
        standard Indian license plate format: 2 Letters, 2 Numbers, 1-2 Letters, 4 Numbers.
        """
        # Remove spaces and non-alphanumeric chars, convert to uppercase
        text = "".join(c.upper() for c in text if c.isalnum())
        
        # If it's not a standard 9 or 10 character Indian plate, just return it
        if len(text) not in [9, 10]:
            return text
            
        char_to_num = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'J':'3', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8'}
        num_to_char = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B'}
        
        corrected = ""
        for i, char in enumerate(text):
            if i < 2:
                # First 2 must be Letters (State Code)
                corrected += num_to_char.get(char, char)
            elif i >= 2 and i < 4:
                # Next 2 must be Numbers (District Code)
                corrected += char_to_num.get(char, char)
            elif i >= len(text) - 4:
                # Last 4 must be Numbers
                corrected += char_to_num.get(char, char)
            else:
                # Middle chars must be Letters
                corrected += num_to_char.get(char, char)
                
        # Return with standard formatting (e.g., HR 26 FC 2782)
        if len(corrected) == 10:
            return f"{corrected[:2]} {corrected[2:4]} {corrected[4:6]} {corrected[6:]}"
        else:
            return f"{corrected[:2]} {corrected[2:4]} {corrected[4:5]} {corrected[5:]}"

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

                # Perform OCR with EasyOCR (limiting to alphanumeric characters)
                allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ocr_result = self.ocr.readtext(processed_plate, allowlist=allowlist)
                
                plate_text = ""
                for res in ocr_result:
                    try:
                        text = res[1]
                        plate_text += text + " "
                    except Exception:
                        pass

                plate_text = plate_text.strip()
                if plate_text:
                    # Apply auto-correction for Indian plates
                    plate_text = self.correct_indian_plate(plate_text)
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

        # Optimize: Reduce resolution for faster processing on Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Press 'q' to quit.")
        
        frame_count = 0
        skip_frames = 3  # Optimize: Process every 3rd frame (skip 2) to save CPU
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame_count += 1
            # Optimize: Frame skipping
            if frame_count % skip_frames != 0:
                cv2.imshow("License Plate Detection (Camera)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Optimize: Run YOLO with smaller imgsz (320 instead of default 640)
            results = self.detector(frame, imgsz=320, verbose=False)
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
            
            # Run detection on the final clear frame (Optimize: smaller imgsz)
            results = self.detector(final_frame, imgsz=320, verbose=False)
            
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
                    
                    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ocr_result = self.ocr.readtext(processed_plate, allowlist=allowlist)
                    
                    plate_text = ""
                    for res in ocr_result:
                        try:
                            text = res[1]
                            plate_text += text + " "
                        except Exception:
                            pass

                    plate_text = plate_text.strip()
                    if plate_text:
                        # Apply auto-correction for Indian plates
                        plate_text = self.correct_indian_plate(plate_text)
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
