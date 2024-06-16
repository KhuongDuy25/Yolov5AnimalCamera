import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, time

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Tải mô hình từ Roboflow (cập nhật đường dẫn đến mô hình của bạn)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\KhuongDuy\Desktop\yolov5-master\best70fist.pt')

# Đường dẫn tới font hỗ trợ Unicode (Arial Unicode MS)
font_path = "arial.ttf"  # Cập nhật đường dẫn tới file font của bạn
font = ImageFont.truetype(font_path, 32)

# Khởi tạo video capture từ camera
camera_index = 0  # Thường là 0 cho camera mặc định
cap = cv2.VideoCapture(camera_index)

# Kiểm tra xem camera có mở thành công không
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Vòng lặp chính để đọc khung hình từ camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện vật thể
    results = model(frame) #frame là khung hình hiện tại lấy từ video hoặc camera.
    #biến results nhận diện các vật thể trong khung hình và lưu kết quả.

    # Lấy kết quả dưới dạng pandas dataframe ,
    df = results.pandas().xyxy[0] #chuyển kết quả nhận diện từ định dạng của YOLOv5 sang một DataFrame của pandas.
    #DataFrame này chứa thông tin về các vật thể được nhận diện, cột như xmin, ymin, xmax, ymax (tọa độ hộp giới hạn), confidence (độ tin cậy), và name (tên của vật thể).
 
    # Lọc kết quả để chỉ bao gồm ngựa vằn
    zebra_df = df[df['name'] == 'Ngựa vằn']
    
    # Đếm số lượng ngựa vằn
    zebra_count = zebra_df.shape[0]

    # Kiểm tra thời gian hiện tại
    current_time = datetime.now().time()
    start_time = time(12, 0)  # giờ sáng
    end_time = time(22, 0)   # giờ chiều

    # Kiểm tra xem có nằm trong khoảng thời gian "xổng chuồng" không
    is_escape_time = start_time <= current_time <= end_time

    # Tạo chuỗi để hiển thị thông tin
    info_str = f"Ngựa vằn phát hiện được: {zebra_count}\n"
    
    # Kiểm tra số lượng ngựa vằn và hiển thị thông báo nếu thiếu
    required_count = 5
    missing_count = required_count - zebra_count
    if missing_count > 0:
        if is_escape_time:
            info_str += f"Xổng chuồng: {missing_count}\n"
        else:
            info_str += f"Đã ra khỏi chuồng: {missing_count}\n"

    # Chuyển đổi từ định dạng OpenCV sang định dạng PIL để hỗ trợ vẽ văn bản Unicode
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Hiển thị thông tin trên góc màn hình
    x, y = 10, 40
    for line in info_str.strip().split('\n'):
        draw.text((x, y), line, font=font, fill=(255, 0, 0))
        y += 40

    # Vẽ các hộp giới hạn xung quanh ngựa vằn được nhận diện
    for _, row in zebra_df.iterrows():
        xmin, ymin, xmax, ymax, label, confidence = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=2)  # Vẽ hộp giới hạn
        draw.text((xmin, ymin - 30), f"{label} {confidence:.2f}", font=font, fill=(0, 255, 0))  # Vẽ nhãn bằng Tiếng Việt

    # Chuyển đổi lại từ định dạng PIL sang định dạng OpenCV để hiển thị
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Hiển thị kết quả
    cv2.imshow("YOLOv5 Nhận Diện", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng video capture và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
