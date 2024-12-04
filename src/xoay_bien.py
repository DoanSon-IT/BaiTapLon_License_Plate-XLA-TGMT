import cv2
import numpy as np
from ultralytics import YOLO

def align_plate(plate_img):
    """
    Xoay ảnh biển số để chính diện dựa trên phân tích góc của các cạnh"""
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện cạnh
    edges = cv2.Canny(blur, 50, 150)
    
    # Phép biến đổi Hough để phát hiện các đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                             threshold=50,  # Ngưỡng phát hiện điểm giao
                             minLineLength=50,  # Độ dài đường tối thiểu
                             maxLineGap=50)  # Khoảng cách tối đa giữa các điểm trên đường
    
    if lines is None or len(lines) == 0:
        return plate_img
    
    # Tách các góc của đường thẳng ngang và dọc
    horizontal_angles = []
    vertical_angles = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Tính góc của đường thẳng
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Phân loại góc ngang và dọc
            if abs(angle) < 30 or abs(angle) > 150:  # Các đường gần như ngang
                horizontal_angles.append(angle)
            elif abs(angle - 90) < 30 or abs(angle + 90) < 30:  # Các đường gần như dọc
                vertical_angles.append(angle)
    
    # Tính góc trung bình của các đường
    def calculate_median_angle(angles):
        if not angles:
            return 0
        return np.median(angles)
    
    # Chọn góc để xoay
    horizontal_median = calculate_median_angle(horizontal_angles)
    vertical_median = calculate_median_angle(vertical_angles)
    
    # Ưu tiên xoay góc ngang (các đường ngang của biển số)
    angle_to_rotate = horizontal_median
    
    # Giới hạn góc xoay
    angle_to_rotate = np.clip(angle_to_rotate, -15, 15)
    
    # Lấy kích thước ảnh
    (h, w) = plate_img.shape[:2]
    center = (w // 2, h // 2)
    
    # Ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle_to_rotate, 1.0)
    
    # Thực hiện xoay
    rotated = cv2.warpAffine(plate_img, M, (w, h), 
                              flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def detect_plates(image_path, model_path=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\license_plate\best.pt", conf_thres=0.3):
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
            
            # Crop the plate
            cropped_plate = img[y1:y2, x1:x2]
            
            # Rotate the plate
            rotated_plate = align_plate(cropped_plate)
            
            plates.append((cropped_plate, rotated_plate, (x1, y1, x2, y2)))

            # Draw bounding box with a different color on original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Put label text above the bounding box - LARGER TEXT
            cv2.putText(img, label_text, 
                        (x1, y1 - 15),  # Moved up a bit 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0,  # Increased font scale from 0.5 to 1.0
                        (0, 0, 255),  # Red color 
                        3)  # Increased thickness from 2 to 3

    # Create a window to display results
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    
    # If plates are detected, show original and rotated images
    if plates:
        for original_plate, rotated_plate, (x1, y1, x2, y2) in plates:
            # Resize images to make them easier to view
            resized_original = cv2.resize(original_plate, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
            resized_rotated = cv2.resize(rotated_plate, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
            
            # Create a side-by-side comparison
            comparison = np.hstack((resized_original, resized_rotated))
            
            # Show the comparison
            cv2.imshow("Original vs Rotated Plate", comparison)
            cv2.waitKey(0)
    
    # Resize and show the original image with detections
    resized_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Detections", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return plates

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\images\17.jpg"  # Replace with your image path
    detect_plates(image_path)