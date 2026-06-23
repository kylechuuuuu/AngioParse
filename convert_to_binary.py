import os
import cv2
import numpy as np
from glob import glob

input_dir = '/hy-tmp/predict_output'
output_dir = '/hy-tmp/predict_output_binary'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

png_files = glob(os.path.join(input_dir, '*.png'))

print(f"Found {len(png_files)} png files.")

for img_path in png_files:
    img_name = os.path.basename(img_path)
    # 以灰度模式读取
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to load {img_name}")
        continue
        
    # 检查最大值，确认是否是低亮度图
    max_val = np.max(img)
    
    # 二值化：只要大于0就变为255
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, binary_img)
    # print(f"Processed {img_name}, original max val: {max_val}")

print(f"All images processed and saved to {output_dir}")
