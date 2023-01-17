import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

img_hr = np.fromfile("/data/zhuz/nemo/1/0000.raw",np.uint8).reshape(1080,1920,3)
print(img_hr.shape)
cv2.imwrite("1.jpg",img_hr)

img_lr = np.fromfile("/data/jinxinqi/Dataset/SuperResolution/NEMO-Dataset/1/image/240p_512kbps_s0_d300.webm/0000.raw",np.uint8).reshape(240,426,3)
img_lr = img_lr[:, :,[2, 1,0]]
print(img_lr.shape)
cv2.imwrite("2.jpg",img_lr)

lr_image = transforms.functional.to_tensor(img_lr)
hr_image = transforms.functional.to_tensor(img_hr)

save_image(hr_image.clamp(0, 1), "hr.png")
save_image(lr_image.clamp(0, 1), "lr.png")


