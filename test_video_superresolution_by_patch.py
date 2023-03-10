import warnings
import argparse
import importlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import save_image

import common.meters
import common.modes
import common.metrics

from utils.estimate import test
from utils import logging_tool
from collections import OrderedDict

import models
from utils import attr_extractor, loss_printer
from loss_config import update_weight
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random
from models.single_image_model import Result_Model
from models.naive_multi_model_easy import Naive_model
from models.basicvsr_arch import BasicVSR
from common.metrics import psnr, psnr_y, ssim


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def total_variation(img):
    #img shape B,N,C,H,W
    B,N,C,H,W= img.shape
    img = img.view(B*N,C,H,W)
    
    img_ = F.pad(img,(0,1,0,1),'replicate')
    img_h = img_[:,:,1:,:-1]
    img_w = img_[:,:,:-1,1:]
    tv = torch.sum(torch.abs(img_h-img)+torch.abs(img_w-img),dim=[-1,-2,-3])
    
    return tv

def time_variation(img):
    B,N,C,H,W= img.shape
    
    img_1 = img[:,1:,:,:,:]
    img_0 = img[:,:-1,:,:,:]
    tv_ = torch.sum(torch.abs(img_1-img_0),dim=[-3,-2,-1])
    
    # tv shape (B,N-1)
    tv = torch.zeros((B,N))
    tv[:,:-1] += tv_
    tv[:,1:] += tv_
    tv[:,0] = tv[:,0]*2
    tv[:,-1] = tv[:,-1]*2
    
    return tv.view(B*N)


def motion_vector_dis(img):
    pass

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


def train(model,
          optimizer,
          train_data_loader,
          criterions,
          writer,
          params,
          epoch,
          device):
    """
    Train model
    """

    loss_meter = common.meters.AverageMeter()
    time_meter = common.meters.TimeMeter()
    block_meter = common.meters.AverageMeter()
    model.train()

    nb = len(train_data_loader)
    losses = {}

    for batch_idx, (lr, hr) in enumerate(train_data_loader):

        hr_label = hr.clone().detach().to(device, non_blocking=True)
        lr = lr.to(device, non_blocking=True)

        total_batches = (epoch - 1) * nb + batch_idx
        
        # Train
        optimizer.zero_grad()
        sr = model(lr)
        loss = 0

        loss_sr_l1 = params.weight_sr_l1 * criterions['l1'](sr, hr_label)
        losses['l1'] = loss_sr_l1
        loss += loss_sr_l1

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        time_meter.update()
        loss_meter.update(loss.item(), sr.size(0))

        if (batch_idx % params.log_steps == 0) and device == 0:
            time_meter.complete_time(nb - batch_idx)

            writer.add_scalar('training_loss', loss.item(), total_batches)
            writer.add_scalar('Loss/l1', loss_sr_l1, total_batches)
            total_epochs = params.epochs
            s = f"## Epoch: {epoch:{' '}{'>'}{2}d}/{total_epochs:d}\t" \
                f"Iters:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb * 100:.2f}%)\t" \
                f"Epoch-est. {time_meter.remain_time}\t" \
                f"Loss: {loss.item():.6f}\t" \
                f"{loss_printer(losses)}"
            print(s)

    if device == 0:
        writer.add_scalar('training_loss_smooth', loss_meter.avg, epoch)
        logging.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{params.epochs} finished.\tLoss: {loss_meter.avg:.6f}")

    # Save image samples per epoch
    if device == 0:
        save_image(sr[:,0,:,:,:], os.path.join(params.job_dir, 'results', f'epoch_{epoch:02d}_output.png'))  # save output
        save_image(hr_label[:,0,:,:,:], os.path.join(params.job_dir, 'results', f'epoch_{epoch:02d}_target.png'))  # save target


def evaluation(model, eval_data_loaders, epoch, writer, device):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param writer: tb_writer
    :param device: index of device
    """
    model.eval()

    patch_h = 64
    patch_w = 64
    overlap_h = 4
    overlap_w = 0
    ##batch_size = 1 
    psnr_list=[]
    bilinear_psnr_list=[]
    space_var_list=[]
    time_var_list=[]
    for eval_data_name, eval_data_loader in eval_data_loaders:
        with torch.no_grad():
            for i, (name, lr, hr) in enumerate(eval_data_loader):
                
                lenh = lr.shape[3]//patch_h
                lenw = lr.shape[4]//patch_w
                start_h=0
                start_w=0

                if torch.cuda.is_available():
                    # hr = hr.to(gpu, non_blocking=True)
                    lr = lr.to('cuda', non_blocking=True)
                if hr.shape[1] == 1:
                    # gray scale to 3 channel
                    hr = torch.stack([hr.squeeze(0), hr.squeeze(0), hr.squeeze(0)], 1)
                    lr = torch.stack([lr.squeeze(0), lr.squeeze(0), lr.squeeze(0)], 1)
                while start_h+patch_h<=lr.shape[3]:
                    while start_w+patch_w<=lr.shape[4]:
                        lr_patch = lr[:,:,:,start_h:start_h+patch_h,start_w:start_w+patch_w]
                        hr_patch = hr[:,:,:,start_h*4:(start_h+patch_h)*4,start_w*4:(start_w+patch_w)*4]
                        print(lr_patch.shape)
                        # start_h+=patch_h-overlap_h
                        # start_w+=patch_w-overlap_w

                        output = model(lr_patch).to('cpu')
                        lr_patch = lr_patch.to('cpu')

                        if len(output) == 1 and output.dim()==5:
                            space_var = total_variation(lr_patch).tolist()
                            space_var_list.extend(space_var)
                            time_var = time_variation(lr_patch).tolist()
                            time_var_list.extend(time_var)
                            

                            for idx in range(lr_patch.shape[1]):
                                output_each = output[:,idx,:,:,:]
                                lr_each = lr_patch[:,idx,:,:,:]
                                hr_each = hr_patch[:,idx,:,:,:]

                                if len(output) == 2:
                                    sr, speed_accu = output_each
                                else:
                                    sr, speed_accu = output_each, None
                            
                                # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
                                baseline = F.interpolate(lr_each,(hr_each.shape[2],hr_each.shape[3]),mode='bilinear')
                                # path_bilinear = f"{f['job_dir']}/eval/bilinear"
                                # os.makedirs(path_bilinear, exist_ok=True)
                                # save_image(baseline.clamp(0, 1), f"{path_bilinear}/{name[0]}_{i}.png")
                                psnr_list.append(psnr(sr,hr_each, shave=4))
                                bilinear_psnr_list.append(psnr(baseline,hr_each, shave=4))
                        start_w+=patch_w-overlap_w
                    start_h+=patch_h-overlap_h
    psnr_list=np.array(psnr_list)
    bilinear_psnr_list=np.array(bilinear_psnr_list)
    space_var_list=np.array(space_var_list)
    time_var_list=np.array(time_var_list)
    print(np.mean(psnr_list),np.mean(bilinear_psnr_list))

    result = np.stack((space_var_list,time_var_list,psnr_list,bilinear_psnr_list),axis=0)
    #4*n
    np.save(os.path.join('run',params.model_type+'_64_64_2'),result)



def trainer_preparation(model, learning_rate, epochs):
    # All optimizer functions and scheduler functions
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(epochs *3 / 5),
                                                           int(epochs * 8 / 10)],
                                               gamma=0.3)
    return optimizer, scheduler


def main(params, logging):
    logging.info(f"Using GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")
    dataset_module = importlib.import_module(f'datasets.{params.dataset}' if params.dataset else 'datasets')

    # Enable cudnn Optimization for static network structure
    torch.backends.cudnn.benchmark = True

    if params.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_seed(local_rank)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        params.world_size = torch.distributed.get_world_size()

    # Set device
    device = local_rank if params.distributed else 0
    update_weight(params=params)
    logging.info(attr_extractor(params), device=device)

    # Create job and tb_writer
    writer = SummaryWriter(params.job_dir) if device == 0 else None



    # Load eval dataset
    if params.eval_datasets:
        eval_datasets = []
        for eval_dataset in params.eval_datasets:
            eval_dataset_module = importlib.import_module(f'datasets.{eval_dataset}')
            eval_datasets.append((eval_dataset, eval_dataset_module.get_dataset(common.modes.EVAL, params)))
    else:
        eval_datasets = [(params.dataset, dataset_module.get_dataset(common.modes.EVAL, params))]

    # Dataloader sampler set to None for single GPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if params.distributed else None
    eval_sampler = None

    if params.debug:
        logging.info('Enable anomaly detect', device=device)
        logging.warning('Debug mode is super slow, set epochs to 1', device=device)
        params.epochs = 1
        torch.autograd.set_detect_anomaly(params.debug)

    # Dataloader


    eval_kwargs = {"num_workers": params.num_data_threads,
                   "batch_size": params.eval_batch_size,
                   'shuffle': False,
                   'drop_last': False,
                   'sampler': eval_sampler}
    eval_data_loaders = [(data_name, DataLoader(dataset=dataset, **eval_kwargs)) for data_name, dataset in
                         eval_datasets]

    # Create Model

    # model = models.get_model(params=params)
    model_type = params.model_type
    if model_type == 'single':
        model = Result_Model(scale=params.scale, filename=params.model_path)
    elif model_type == 'multi':
        model = Naive_model(scale=params.scale, filename=params.model_path,spynet_pretrained='/home/zhuzhui/BasicVSR_PlusPlus/model/spynet_20210409-c6c1bd09.pth')
    elif model_type == 'basic':
        model = BasicVSR(num_feat=24, num_block=7, spynet_path='/home/zhuzhui/BasicVSR_PlusPlus/model/spynet_20210409-c6c1bd09.pth')
    else:
        raise Exception("????????????")
    logging.info(f"\n{model}", is_print=False, device=device)

    if params.distributed:
        logging.info("Distributed Training", device=device)
        params.learning_rate *= params.world_size

    # Loss function
    criterions = OrderedDict()
    criterions['l1'] = L1_Charbonnier_loss().to(device)

    if params.eval_model:
        model.load_state_dict(torch.load(params.eval_model),strict=True)
        # model.load_state_dict(torch.load(params.eval_model)['params'],strict=True)

    if device == 0:
        for folder in ['results', 'weights', 'ckpt', 'eval']:
            folder_path = os.path.join(params.job_dir, folder)
            if os.path.exists(folder_path):
                logging.warning(f'{os.path.join(params.job_dir, folder)} already exists', device=device)
            else:
                os.mkdir(folder_path)
                logging.info(f'Create {folder_path}', device=device)

    # Train
    epoch = 0
    end_epoch = 0

    model.to("cuda")
    evaluation(model, eval_data_loaders, epoch, writer, device=device)



    if not params.eval_only and device == 0:
        writer.close()

    logging.info(f"Finish Training", device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='NAS_MODEL', type=str, required=True,
                        help='model type: NAS_MODEL / BASIC_MODEL')

    parser.add_argument('--dataset', default=None, type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('--job_dir', default=None, type=str, required=True,
                        help='Directory to write checkpoints and export models.')
    parser.add_argument('--model_path', default=None, type=str,
                        help='File path to load checkpoint.')
    parser.add_argument('--model_weight', default=None, type=str,
                        help='path to weight')
    parser.add_argument('--scheduler_type', default='multi_step', type=str,
                        help="LR scheduler type: cosine, multi_step")
    parser.add_argument('--image_batch', default=10, type=int,
                        help=" ")
    parser.add_argument('--val_image_batch', default=100, type=int,
                        help=" ")

    # evaluation
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='Running evaluation only.')
    parser.add_argument('--eval_model', default=None, type=str,
                        help='Path to evaluation model.')
    parser.add_argument('--eval_datasets', default=None, type=str, nargs='+',
                        help='Dataset names for evaluation.')

    # Experiment arguments
    parser.add_argument('--speed_target', default=40, type=float,
                        help='speed target in ms.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of search epochs.')
    parser.add_argument('--width_epochs', default=0, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--finetune_epochs', default=30, type=int,
                        help='Number of fine-tune epochs.')
    parser.add_argument('--log_steps', default=100, type=int,
                        help='Number of steps for training logging.')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='Recognized opt_levels are O0, O1, O2, and O3')

    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume training from ckpt')
    parser.add_argument('--warmup_lr', default=False, action='store_true',
                        help='Using warmup lr')

    # Verbose
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Increasing output verbosity.', )
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug mode. Enable torch anomaly check')

    # for multi GPU
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='Distributed training')

    # Parse arguments
    args, _ = parser.parse_known_args()
    logging = logging_tool.LoggingTool(file_path=args.job_dir, verbose=args.verbose)

    # Update args
    importlib.import_module(f'datasets.{args.dataset}' if args.dataset else 'datasets').update_argparser(parser)
    models.update_argparser(parser)

    # parsing args
    params = parser.parse_args()

    main(params, logging)
