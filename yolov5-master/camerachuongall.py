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

# Số lượng yêu cầu của từng loài động vật
required_counts = {
    'Chuột lang nước': 2, 'Bò': 2, 'Hươu': 2, 'Voi': 2, 'Hồng hạc': 2,
    'Hươu cao cổ': 2, 'Báo đốm': 2, 'Kangaroo': 2, 'Sư tử': 2, 'Vẹt': 2,
    'Chim cánh cụt': 2, 'Tê giác': 2, 'Cừu': 2, 'Hổ': 2, 'Rùa': 2, 'Ngựa vằn': 4
}

# Vòng lặp chính để đọc khung hình từ camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện vật thể
    results = model(frame)

    # Lấy kết quả dưới dạng pandas dataframe
    df = results.pandas().xyxy[0]

    # Lọc kết quả để chỉ bao gồm các loài động vật trong danh sách
    filtered_df = df[df['name'].isin(required_counts.keys())]

    # Đếm số lượng các loài động vật
    animal_counts = filtered_df['name'].value_counts().to_dict()

    # Kiểm tra thời gian hiện tại
    current_time = datetime.now().time()
    start_time = time(17, 0)  # giờ sáng
    end_time = time(7, 0)   # giờ chiều

    # Kiểm tra xem có nằm trong khoảng thời gian "xổng chuồng" không
    is_escape_time = start_time <= current_time <= end_time

    # Tạo chuỗi để hiển thị thông tin
    info_str = ""

    for animal, required_count in required_counts.items():
        if animal in animal_counts:
            count = animal_counts[animal]
            missing_count = required_count - count
            if missing_count > 0:
                if is_escape_time:
                    info_str += f"{animal}: {count} (Xổng chuồng): {missing_count}\n"
                else:
                    info_str += f"{animal}: {count} (Đã ra khỏi chuồng): {missing_count}\n"
            else:
                info_str += f"{animal}: {count}\n"

    # Chuyển đổi từ định dạng OpenCV sang định dạng PIL để hỗ trợ vẽ văn bản Unicode
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Hiển thị thông tin trên góc màn hình
    x, y = 10, 40
    for line in info_str.strip().split('\n'):
        draw.text((x, y), line, font=font, fill=(255, 0, 0))
        y += 40

    # Vẽ các hộp giới hạn xung quanh các loài động vật được nhận diện
    for _, row in filtered_df.iterrows():
        xmin, ymin, xmax, ymax, label, confidence = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=2)  # Vẽ hộp giới hạn
        draw.text((xmin, ymin - 30), f"{label} {confidence:.2f}", font=font, fill=(0, 255, 0))  # Vẽ nhãn bằng Tiếng Việt

    # Chuyển đổi lại từ định dạng
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Hiển thị kết quả
    cv2.imshow("YOLOv5 Nhận Diện", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng video capture và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()