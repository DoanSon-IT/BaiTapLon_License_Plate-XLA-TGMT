import cv2
from ultralytics import YOLO

def detect_plates(image_path, model_path=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\license_plate\best.pt", conf_thres=0.1):

    model = YOLO(model_path)
    img = cv2.imread(image_path)

    results = model.predict(source=img, conf=conf_thres)

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and label
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            
            # Create label text with label and confidence
            label_text = f"{label} {conf:.2f}"
            
            cropped_plate = img[y1:y2, x1:x2]
            plates.append((cropped_plate, (x1, y1, x2, y2)))

            # Draw bounding box with a different color
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Put label text above the bounding box - LARGER TEXT
            cv2.putText(img, label_text, 
                        (x1, y1 - 15),  # Moved up a bit 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0,  # Increased font scale from 0.5 to 1.0
                        (0, 0, 255),  # Red color 
                        3)  # Increased thickness from 2 to 3

    # Resize image to show (50% of original size)
    resized_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Detections", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return plates