import os

dir = '/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/runs/wdsr_b_x4_16_24_Jan20_13_34_38/eval/bilinear'

files = sorted(os.listdir(dir))

i=0
for file in files:
    os.rename(os.path.join(dir,file),os.path.join(dir,str(i).zfill(4)+'.png'))
    i+=1