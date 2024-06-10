import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Tải mô hình từ Roboflow (cập nhật đường dẫn đến mô hình của bạn)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\KhuongDuy\Desktop\yolov5-master\best70fist.pt')

# Chuẩn bị hình ảnh
image_path = r'C:\Users\KhuongDuy\Desktop\yolov5-master\data\test\chet-cuoi-canh-ngua-van-va-huou-cao-co-thi-chay.jpg'
img = cv2.imread(image_path)

# Nhận diện vật thể
results = model(img)

# Lấy kết quả dưới dạng pandas dataframe
df = results.pandas().xyxy[0]

# Danh sách các loài động vật cần đếm
animal_names = [
    'Chuột lang nước', 'Bò', 'Hươu', 'Voi', 'Hồng hạc', 'Hươu cao cổ', 'Báo đốm', 
    'Kangaroo', 'Sư tử', 'Vẹt', 'Chim cánh cụt', 'Tê giác', 'Cừu', 'Hổ', 'Rùa', 'Ngựa vằn'
]

# Lọc kết quả để chỉ bao gồm các loài động vật trong danh sách
filtered_df = df[df['name'].isin(animal_names)]

# Đếm số lượng các loài động vật
animal_counts = filtered_df['name'].value_counts()

# Tạo một chuỗi để hiển thị số lượng các loài động vật bằng Tiếng Việt
animal_counts_str = '\n'.join([f"{name}: {count}" for name, count in animal_counts.items()])

# Chuyển đổi từ định dạng OpenCV sang định dạng PIL để hỗ trợ vẽ văn bản Unicode
img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
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
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Hiển thị kết quả
cv2.imshow("YOLOv5 Nhận Diện", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
