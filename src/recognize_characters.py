import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO

# Đường dẫn mặc định đến mô hình nhận diện ký tự
CHAR_MODEL_PATH = r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\char\best.pt"

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

def sort_boxes(detected_chars, max_line_gap= 150):
    """
    Sắp xếp bounding box ký tự theo hàng và cột (trái qua phải, trên xuống dưới),
    với khả năng phân nhóm thành 2 hàng cho biển số xe máy.
    """
    if not detected_chars:  # Nếu không có ký tự nào được phát hiện
        return []

    # Sắp xếp theo y trước, x sau
    detected_chars.sort(key=lambda box: (box[1], box[0]))

    lines = []
    current_line = [detected_chars[0]]

    for i in range(1, len(detected_chars)):
        # Kiểm tra sự chênh lệch giữa vị trí y của ký tự hiện tại và ký tự trước đó
        if abs(detected_chars[i][1] - current_line[0][1]) <= max_line_gap:
            current_line.append(detected_chars[i])
        else:
            # Nếu chênh lệch quá lớn, xác định đây là một hàng mới
            lines.append(sorted(current_line, key=lambda box: box[0]))
            current_line = [detected_chars[i]]

    # Đảm bảo rằng các ký tự cuối cùng được thêm vào
    lines.append(sorted(current_line, key=lambda box: box[0]))

    # Bây giờ chia thành các nhóm, tùy thuộc vào số lượng dòng
    # Giả sử biển số xe máy có hai hàng, ta kiểm tra các dòng có ít ký tự hơn là hàng trên
    sorted_chars = []
    if len(lines) > 1:  # Nếu có 2 hàng
        first_line = lines[0]
        second_line = lines[1]
        
        # Giả sử dòng đầu tiên có ít ký tự hơn (đặc trưng của biển số xe máy)
        sorted_chars = first_line + second_line
    else:
        sorted_chars = [char for line in lines for char in line]

    return sorted_chars, lines

def recognize_characters(plate_img, image_name, output_dir, conf_thres=0.5):
    # Resize ảnh biển số về 640x640
    plate_img_resized = cv2.resize(plate_img, (640, 640), interpolation=cv2.INTER_AREA)

    # Load mô hình YOLO nhận diện ký tự
    char_model = YOLO(CHAR_MODEL_PATH)

    # Dự đoán ký tự trên biển số
    results = char_model.predict(source=plate_img_resized, conf=conf_thres)
    
    # Tạo bản sao của ảnh để vẽ bounding box
    result_plate = plate_img_resized.copy()
    
    detected_chars = []
    
    for result in results:
        for box in result.boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Lấy độ tin cậy và nhãn
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = char_model.names[cls]
            
            # Vẽ bounding box
            cv2.rectangle(result_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ghi nhãn và độ tin cậy
            text = f"{label} {conf:.2f}"
            cv2.putText(result_plate, text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Lưu thông tin ký tự
            detected_chars.append((x1, y1, x2, y2, label, conf))

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu ảnh có bounding box
    output_path = os.path.join(output_dir, f"{image_name}_chars.jpg")
    cv2.imwrite(output_path, result_plate)
    
    return detected_chars

def save_and_show_images(original_img, plates, output_dir):
    # Tạo file CSV để lưu thông tin ký tự
    csv_path = os.path.join(output_dir, "characters.csv")
    csv_headers = ["Image Name", "Complete Text"]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)

        for idx, (plate_img, bbox) in enumerate(plates, 1):
            x1, y1, x2, y2 = bbox
            cropped_plate = original_img[y1:y2, x1:x2]
            
            aligned_plate = align_plate(cropped_plate)
            
            

            # Đặt tên ảnh
            plate_name = f"plate_{idx}"

            # Tạo thư mục con cho từng ảnh gốc
            plate_output_dir = os.path.join(output_dir, plate_name)
            os.makedirs(plate_output_dir, exist_ok=True)

            try:
                # Nhận diện ký tự và lưu ảnh
                detected_chars = recognize_characters(aligned_plate, plate_name, plate_output_dir)

                # Sắp xếp các ký tự
                sorted_chars, lines = sort_boxes(detected_chars)

                # Kết hợp ký tự thành chuỗi biển số
                complete_text = " ".join("".join(char[4] for char in line) for line in lines)

                # Lưu thông tin vào CSV
                csv_writer.writerow([plate_name, complete_text])

                print(f"Plate {idx}: {complete_text}")

                # Đọc và hiển thị ảnh có bounding box
                output_path = os.path.join(plate_output_dir, f"{plate_name}_chars.jpg")
                
                if os.path.exists(output_path):
                    labeled_img = cv2.imread(output_path)
                    
                    # Resize hình ảnh để hiển thị
                    labeled_img_resized = cv2.resize(labeled_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

                    # Hiển thị hình ảnh
                    cv2.imshow(f"Characters in Plate {idx}", labeled_img_resized)
                else:
                    print(f"Image file not found: {output_path}")
            
            except Exception as e:
                print(f"Error processing plate {idx}: {e}")
                continue

    # Chờ người dùng tắt cửa sổ hiển thị
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Character information saved to {csv_path}")
