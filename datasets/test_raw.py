import numpy as np
import cv2

img = np.fromfile("/data/zhuz/nemo/1/0000.raw",np.uint8).reshape(1080,1920,3)
print(img.shape)
cv2.imwrite("1.jpg",img)

img = np.fromfile("/data/jinxinqi/Dataset/SuperResolution/NEMO-Dataset/1/image/240p_512kbps_s0_d300.webm/0000.raw",np.uint8).reshape(240,426,3)
img = img[:, :,[2, 1,0]]
print(img.shape)
cv2.imwrite("2.jpg",img)


