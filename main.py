import os
import cv2
from src.detect_plate import detect_plates
from src.recognize_characters import save_and_show_images

def main():
    # Đường dẫn ảnh đầu vào
    image_path = r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\\test_img_video\images\1.jpg"

    # Đường dẫn thư mục lưu kết quả
    output_dir = r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\output_img_video\output_img"

    # Đọc ảnh gốc
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn ảnh.")
        return

    # Phát hiện biển số trên ảnh
    print("Đang phát hiện biển số...")
    plates = detect_plates(image_path)
    if not plates:
        print("Không tìm thấy biển số trên ảnh.")
        return

    # Lưu và hiển thị kết quả nhận diện ký tự trên biển số
    print("Đang nhận diện ký tự trên biển số...")
    save_and_show_images(original_img, plates, output_dir)

    print(f"Kết quả đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    main()
