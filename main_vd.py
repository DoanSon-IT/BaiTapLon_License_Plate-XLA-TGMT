import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO

# Tạo mapping từ class ID sang ký tự
class_to_char = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
    19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
    28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"
}

def align_plate(plate_img):
    """
    Xoay ảnh biển số để chính diện dựa trên phân tích góc của các cạnh
    """
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

def sort_boxes(detected_chars, max_line_gap=50):
    """
    Sắp xếp bounding box ký tự theo hàng và cột (trái qua phải, trên xuống dưới),
    với khả năng phân nhóm thành 2 hàng cho biển số xe máy.
    """
    if not detected_chars:  # Nếu không có ký tự nào được phát hiện
        return [], []

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

def process_video(video_path, output_directory, conf_plate_thresh=0.3, conf_char_thresh=0.5):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_directory, exist_ok=True)
    
    # Đường dẫn đầy đủ cho các file đầu ra
    output_video_path = os.path.join(output_directory, "video4.mp4")
    output_csv_path = os.path.join(output_directory, "license_plate_results4.csv")

    # Load models
    model_plate = YOLO(r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\license_plate\best.pt")
    model_chars = YOLO(r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\char\best.pt")

    # Mở video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Tạo CSV để lưu kết quả
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "License_Plate_Text", "Confidence"])

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Nhận diện biển số
            results_plate = model_plate.predict(source=frame, conf=conf_plate_thresh, save=False, save_txt=False)[0]
            
            for plate_box in results_plate.boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, plate_box[:4])
                plate_conf = float(results_plate.boxes.conf[results_plate.boxes.xyxy.tolist().index(plate_box)].item())
                
                # Cắt và chỉnh sửa biển số
                cropped_plate = frame[y1:y2, x1:x2]
                aligned_plate = align_plate(cropped_plate)

                # Resize biển số cho nhận diện ký tự
                aligned_plate_resized = cv2.resize(aligned_plate, (640, 640), interpolation=cv2.INTER_AREA)

                # Nhận diện ký tự
                results_chars = model_chars.predict(source=aligned_plate_resized, conf=conf_char_thresh, save=False, save_txt=False)[0]
                
                chars = []
                for char_box, char_cls in zip(results_chars.boxes.xyxy.tolist(), results_chars.boxes.cls.tolist()):
                    # Điều chỉnh tọa độ ký tự để nằm trong vùng biển số
                    cx1, cy1, cx2, cy2 = map(int, char_box[:4])
                    
                    # Scale lại tọa độ để phù hợp với kích thước biển số gốc
                    scaled_cx1 = int(cx1 * (x2 - x1) / 640) + x1
                    scaled_cy1 = int(cy1 * (y2 - y1) / 640) + y1
                    scaled_cx2 = int(cx2 * (x2 - x1) / 640) + x1
                    scaled_cy2 = int(cy2 * (y2 - y1) / 640) + y1
                    
                    label = class_to_char[int(char_cls)]
                    char_conf = float(results_chars.boxes.conf[results_chars.boxes.xyxy.tolist().index(char_box)].item())
                    chars.append((scaled_cx1, scaled_cy1, scaled_cx2, scaled_cy2, label, char_conf))

                # Sắp xếp ký tự
                sorted_chars, lines = sort_boxes(chars)

                # Tạo chuỗi ký tự
                license_plate_text = " ".join("".join(char[4] for char in line) for line in lines)

                # Ghi kết quả vào CSV
                writer.writerow([frame_index, license_plate_text, plate_conf])

                # Vẽ bounding box biển số
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 
                            f"{license_plate_text} ({plate_conf:.2f})", 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2)

                # Vẽ bounding box từng ký tự TRONG biển số
                for char in sorted_chars:
                    cv2.rectangle(frame, (char[0], char[1]), (char[2], char[3]), (255, 0, 0), 1)
                    # Thêm nhãn ký tự
                    cv2.putText(frame, char[4], 
                                (char[0], char[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 0, 0), 1)

            # Ghi khung hình
            out.write(frame)

            # Hiển thị (tùy chọn)
            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to {output_video_path}")
    print(f"Results saved to {output_csv_path}")

# Sử dụng hàm xử lý video
process_video(
    video_path=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\test_img_video\video\video2.mp4",
    output_directory=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\output_img_video\output_video",
    conf_plate_thresh=0.3,   # Ngưỡng nhận diện biển số
    conf_char_thresh=0.1     # Ngưỡng nhận diện ký tự
)