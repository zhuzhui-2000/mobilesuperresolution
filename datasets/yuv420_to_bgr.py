import cv2
import numpy as np
import os

def sr_convert_yuv_to_rgb(yuv_dir,raw_dir,index: int, height=1080, width=1920):
    index = str(index).zfill(4)
    with open(os.path.join(yuv_dir,"{}.y".format(index)), "rb") as f:
        y=np.frombuffer(f.read(height*width), dtype=np.uint8).reshape(height, width)
    with open(os.path.join(yuv_dir,"{}.u".format(index)), "rb") as f:
        u=np.frombuffer(f.read(height*width//4), dtype=np.uint8).reshape(height//2, width//2)
    with open(os.path.join(yuv_dir,"{}.v".format(index)), "rb") as f:
        v=np.frombuffer(f.read(height*width//4), dtype=np.uint8).reshape(height//2, width//2)
    u = cv2.resize(u, (width, height))
    v = cv2.resize(v, (width, height))
    yvu = cv2.merge((y, v, u))
    bgr = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR)
    # Alternatively, use the rgb format:
    #rgb = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2RGB)
    bgr.tofile(os.path.join(raw_dir,"{}.raw".format(index)))
    # To read [index].raw:
    # np.fromfile("{}.raw".format(index), dtype=np.uint8).reshape(height, width, 3)

if __name__ == "__main__":
    video_idx = 1
    
    nemo_dir = '/data/jinxinqi/Dataset/SuperResolution/NEMO-Dataset'
    yuv_dir = nemo_dir +'/' + str(video_idx)+'/image/2160p_12000kbps_s0_d300.webm'
    raw_dir = '/data/zhuz/nemo/' + str(video_idx)
    os.makedirs(raw_dir,exist_ok=True)
    files = sorted(os.listdir(yuv_dir))

    idx_list=[]
    for file_name in files:
        frame_idx = int(file_name.split(".")[0])
        if frame_idx not in idx_list:
            idx_list.append(frame_idx)
    
    for idx in idx_list:
        sr_convert_yuv_to_rgb(yuv_dir,raw_dir,idx)
