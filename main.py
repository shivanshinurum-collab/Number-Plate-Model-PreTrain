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

                cv2.imwrite(f"plate_{i}.jpg", plate)

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

if __name__ == "__main__":
    IMAGE_PATH = ["test/car1.png","test/car2.png","test/car3.png","test/car4.png","test/car5.png","test/bike1.png"]
    # Using the better named detector model
    MODEL_PATH = "license_plate_detector.pt"
    # MODEL_PATH = "best.pt"
    

    for img in IMAGE_PATH:
        try:
            reader = LicensePlateReader(model_path=MODEL_PATH)
            texts = reader.read_plate(image_path=img)
        
            print("\n--- Final Results ---")
            for idx, text in enumerate(texts):
                print(f"Plate {idx + 1}: {text}")
            
        except Exception as e:
            print(f"Error occurred: {e}")


# if __name__ == "__main__":
#     IMAGE_PATH = "test/car1.png"
#     # Using the better named detector model
#     MODEL_PATH = "license_plate_detector.pt"
    
#     try:
#         reader = LicensePlateReader(model_path=MODEL_PATH)
#         texts = reader.read_plate(image_path=IMAGE_PATH)
        
#         print("\n--- Final Results ---")
#         for idx, text in enumerate(texts):
#             print(f"Plate {idx + 1}: {text}")
            
#     except Exception as e:
#         print(f"Error occurred: {e}")
