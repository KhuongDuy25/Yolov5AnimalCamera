import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Tải mô hình yolov5 (cập nhật đường dẫn đến mô hình của bạn)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\KhuongDuy\Desktop\yolov5-master\yolov5s.pt')

# Danh sách các loài động vật cần đếm
animal_names = [ 
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có mở thành công không
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Đường dẫn tới font hỗ trợ Unicode (Arial Unicode MS)
font_path = "arial.ttf"  # Cập nhật đường dẫn tới file font của bạn
font = ImageFont.truetype(font_path, 32)

# Vòng lặp chính để đọc khung hình từ webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Nhận diện vật thể
    results = model(frame)

    # Lấy kết quả dưới dạng pandas dataframe
    df = results.pandas().xyxy[0]

    # Lọc kết quả để chỉ bao gồm các loài động vật trong danh sách
    filtered_df = df[df['name'].isin(animal_names)]

    # Đếm số lượng các loài động vật
    animal_counts = filtered_df['name'].value_counts()

    # Tạo một chuỗi để hiển thị số lượng các loài động vật bằng Tiếng Việt
    animal_counts_str = '\n'.join([f"{name}: {count}" for name, count in animal_counts.items()])

    # Chuyển đổi từ định dạng OpenCV sang định dạng PIL để hỗ trợ vẽ văn bản Unicode
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
  
    font = ImageFont.truetype("arial.ttf", 32)

    # Hiển thị số lượng trên góc màn hình bằng Tiếng Việt
    x, y = 10, 40
    for line in animal_counts_str.split('\n'):
        draw.text((x, y), line, font=font, fill=(255, 0, 0))
        y += 40

    # Vẽ các hộp giới hạn xung quanh các vật thể được nhận diện
    for _, row in filtered_df.iterrows():
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

# Giải phóng webcam và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
