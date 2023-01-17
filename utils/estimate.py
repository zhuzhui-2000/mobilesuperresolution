import os
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from common.metrics import psnr, psnr_y, ssim
from torchvision.utils import save_image
from torch.nn.functional import interpolate
import time

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-12
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def test(dataloader: torch.utils.data.DataLoader, model: nn.Module, gpu: torch.device, f: dict, save=True):
    # switch eval mode.
    model.eval()
    total_psnr_value = 0.
    total_psnr_y_value = 0.
    total_ssim_value = 0.
    total_biniliear_psnr = 0.
    total_biniliear_ssim = 0.
    
    total = 0
    criterions= L1_Charbonnier_loss().to('cuda')
    l=0
    with torch.no_grad():
        for i, (name, lr, hr) in enumerate(dataloader):
            if f['epoch'] % 5 >0 and i>20:
                break
            t=time.time()
            if torch.cuda.is_available():
                # hr = hr.to(gpu, non_blocking=True)
                lr = lr.to(gpu, non_blocking=True)
            if hr.shape[1] == 1:
                # gray scale to 3 channel
                hr = torch.stack([hr.squeeze(0), hr.squeeze(0), hr.squeeze(0)], 1)
                lr = torch.stack([lr.squeeze(0), lr.squeeze(0), lr.squeeze(0)], 1)

            output = model(lr,hr.shape[3],hr.shape[4]).to('cpu')
            print(output.shape,hr.shape,time.time()-t)
            lr = lr.to('cpu')

            if len(output) == 1 and output.dim()==5:
                total+=output.shape[1]
                for i in range(lr.shape[1]):
                    
                    if len(output) == 1 and output.dim()==5:
                        
                        output_each = output[:,i,:,:,:]
                        lr_each = lr[:,i,:,:,:]
                        hr_each = hr[:,i,:,:,:]

                    if len(output) == 2:
                        sr, speed_accu = output_each
                    else:
                        sr, speed_accu = output_each, None
                    path = f"{f['job_dir']}/eval/{f['eval_data_name']}"
                    
                    l+=criterions(sr,hr_each)
                    os.makedirs(path, exist_ok=True)
                    path_bilinear = f"{f['job_dir']}/eval/bilinear"
                    os.makedirs(path_bilinear, exist_ok=True)
                    path_hr = f"{f['job_dir']}/eval/hr"
                    os.makedirs(path_hr, exist_ok=True)

                    
                    scale = model.scale if not hasattr(model, 'module') else model.module.scale
                    # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
                    baseline = interpolate(lr_each,(hr_each.shape[2],hr_each.shape[3]),mode='bilinear')
                    
                    if save:
                        save_image(sr.clamp(0, 1), "{0}/{1}{2:0>3d}.png".format(path,name[0],i))
                        save_image(baseline.clamp(0, 1), "{0}/{1}{2:0>3d}.png".format(path_bilinear,name[0],i))
                        save_image(hr_each.clamp(0, 1), "{0}/{1}{2:0>3d}.png".format(path_hr,name[0],i))
                    # path_bilinear = f"{f['job_dir']}/eval/bilinear"
                    # os.makedirs(path_bilinear, exist_ok=True)
                    # save_image(baseline.clamp(0, 1), f"{path_bilinear}/{name[0]}_{i}.png")
                if len(output) == 2:
                    sr, speed_accu = output
                else:
                    sr, speed_accu = output, None
                lr = lr.squeeze()
                hr = hr.squeeze()
                sr = sr.squeeze()
                baseline = interpolate(lr,(hr.shape[2],hr.shape[3]),mode='bilinear')
                
                total_biniliear_psnr += psnr(baseline,hr, shave=4)
                total_psnr_y_value += psnr_y(sr, hr, shave=4)
                total_psnr_value += psnr(sr, hr, shave=4)
                # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
                total_ssim_value += ssim(sr, hr, shave=scale)
                total_biniliear_ssim += ssim(baseline, hr, shave=scale)
                
            else:
                total+=lr.shape[0]
                if len(output) == 2:
                    sr, speed_accu = output
                else:
                    sr, speed_accu = output, None
                path = f"{f['job_dir']}/eval/{f['eval_data_name']}"
                os.makedirs(path, exist_ok=True)
                if save:
                    save_image(sr.clamp(0, 1), f"{path}/{name[0]}.png")
                scale = model.scale if not hasattr(model, 'module') else model.module.scale
                baseline = interpolate(lr,(hr.shape[2],hr.shape[3]),mode='bilinear', align_corners=True)
                total_biniliear_psnr += psnr(baseline,hr, shave=scale + 6)
                # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
                total_psnr_y_value += psnr_y(sr, hr, shave=scale)
                total_psnr_value += psnr(sr, hr, shave=scale + 6)
                # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
                total_ssim_value += ssim(sr, hr, shave=scale)
                total_biniliear_ssim += ssim(baseline, hr_each, shave=scale)
    print("test loss:",l/total)
            # Shave boarder
    out = total_psnr_value / total, total_psnr_y_value / total, total_ssim_value / total, speed_accu , total_biniliear_psnr / total, total_biniliear_ssim / total
    return out
