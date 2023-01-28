hr_path="/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/runs/wdsr_b_x4_16_24_Jan20_13_06_54/eval/hr/%04d.png"
lr_path="/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/runs/wdsr_b_x4_16_24_Jan20_13_34_38/eval/nemo/%04d.png"

ffmpeg -start_number 0 -i ${lr_path} -start_number 0 \
-i ${hr_path}  -lavfi libvmaf  -f null –