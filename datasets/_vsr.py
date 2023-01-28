"""Image Super-resolution dataset."""

import os
import random
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte
import time
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import cv2
import common.modes
import common.io
from common.images import imresize
import datasets


def update_argparser(parser):
    datasets.update_argparser(parser)
    parser.add_argument(
        '--scale',
        help='Scale factor for image super-resolution.',
        default=2,
        type=int)
    parser.add_argument(
        '--lr_patch_size',
        help='Number of pixels in height or width of LR patches.',
        default=48,
        type=int)
    parser.add_argument(
        '--ignored_boundary_size',
        help='Number of ignored boundary pixels of LR patches.',
        default=2,
        type=int)
    parser.add_argument(
        '--num_patches',
        help='Number of sampling patches per image for training.',
        default=100,
        type=int)
    parser.set_defaults(
        train_batch_size=16,
        eval_batch_size=1,
        image_mean=0.5,
    )


class VideoSuperResolutionDataset(data.Dataset):

    def __init__(self, mode, params, lr_files, hr_files, image_batch=10):
        super(VideoSuperResolutionDataset, self).__init__()
        self.mode = mode
        self.params = params
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.image_batch = image_batch

    def __getitem__(self, index):
        
        if self.mode == common.modes.PREDICT:
            lr_image = np.asarray(Image.open(self.lr_files[index][1]))
            lr_image = transforms.functional.to_tensor(lr_image)
            return lr_image, self.hr_files[index][0]

        if self.mode == common.modes.TRAIN:
            index = index // self.params.num_patches
        
        lr_image_list, hr_image_list = self._load_item(index) #numpy list
        lr_image_list_ = []
        hr_image_list_ = []
        # lr_image, hr_image = self._sample_patch(lr_image, hr_image)
        
        p1 = random.random()
        p2 = random.random()
        
        if lr_image_list[0].shape[0]<=68:
            x = 0
        else:
            x = random.randrange(
            self.params.ignored_boundary_size, lr_image_list[0].shape[0] -
                                            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        y = random.randrange(
            self.params.ignored_boundary_size, lr_image_list[0].shape[1] -
                                            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        for (idx,(lr_image,hr_image)) in enumerate(zip(lr_image_list,hr_image_list)):
            if self.mode == common.modes.TRAIN:
                
                if self.params.train_sample_patch:
                    # lr_image, hr_image = self._augment(lr_image, hr_image,p1,p2)
                    lr_image, hr_image = self._sample_patch(lr_image,hr_image,x,y)
            lr_image = np.ascontiguousarray(lr_image)
            hr_image = np.ascontiguousarray(hr_image)
            
            lr_image = transforms.functional.to_tensor(lr_image)
            hr_image = transforms.functional.to_tensor(hr_image)
            
            lr_image_list_.append(lr_image)
            hr_image_list_.append(hr_image)

        # for lr_image in lr_image_list:
        #     lr_image = np.ascontiguousarray(lr_image)
        #     lr_image = transforms.functional.to_tensor(lr_image)
        #     lr_image_list_.append(lr_image)
        # for hr_image in hr_image_list:
        #     hr_image = np.ascontiguousarray(hr_image)
        #     hr_image = transforms.functional.to_tensor(hr_image)
        #     hr_image_list_.append(hr_image)

        lr_image = torch.stack(lr_image_list_)
        hr_image = torch.stack(hr_image_list_)
        
        if self.mode == common.modes.TRAIN:
            lr_image, hr_image = self._augment(lr_image, hr_image,p1,p2)
        # print(lr_image.shape,hr_image.shape)


        
        if self.mode == common.modes.TRAIN:
            return lr_image, hr_image
        elif self.mode == common.modes.EVAL:

            p=str.split(os.path.splitext(self.lr_files[index][0])[0],"/")
            save_path = p[-2]+p[-1]
            # save_path = os.path.splitext(self.lr_files[index][0])[-2]+'_'+os.path.splitext(self.lr_files[index][0])[-1]
            return save_path, lr_image, hr_image

    def _load_item(self, index):
        lr_image_list=[]
        hr_image_list=[]
        
        for (lr_path,hr_path) in zip(self.lr_files[index],self.hr_files[index]):
            
            lr_image = np.asarray(Image.open(lr_path))
            hr_image = np.asarray(Image.open(hr_path))
            lr_image = np.ascontiguousarray(lr_image)
            hr_image = np.ascontiguousarray(hr_image)
            lr_image = transforms.functional.to_tensor(lr_image)
            hr_image = transforms.functional.to_tensor(hr_image)
            lr_image_list.append(lr_image)
            hr_image_list.append(hr_image)
        
        lr_image = torch.stack(lr_image_list)
        hr_image = torch.stack(hr_image_list)
        return lr_image_list, hr_image_list

    def _sample_patch(self, lr_image, hr_image,x,y):
        
        if self.mode == common.modes.TRAIN:
            # sample patch while training

            lr_image = lr_image[x:x + self.params.lr_patch_size, y:y +
                                                                   self.params.lr_patch_size,:]
            hr_image = hr_image[x *
                                self.params.scale:(x + self.params.lr_patch_size) *
                                                  self.params.scale, y *
                                                                     self.params.scale:(y + self.params.lr_patch_size) *
                                                                                       self.params.scale,:]
        # else:
        #     hr_image = hr_image[:lr_image.shape[2] *
        #                          self.params.scale, :lr_image.shape[3] *
        #                                              self.params.scale]
        return lr_image, hr_image

    def _augment(self, lr_image, hr_image,p1,p2):

        if self.mode == common.modes.TRAIN:
            
            for idx in range(lr_image.shape[0]):
            # augmentation while training
                if p1 < 0.5:
                    lr_image[idx,:,:,:] = transforms.RandomHorizontalFlip(p=1)(lr_image[idx,:,:,:])
                    hr_image[idx,:,:,:] = transforms.RandomHorizontalFlip(p=1)(hr_image[idx,:,:,:])
                if p2 < 0.5:
                    lr_image[idx,:,:,:] = transforms.RandomVerticalFlip(p=1)(lr_image[idx,:,:,:])
                    hr_image[idx,:,:,:] = transforms.RandomVerticalFlip(p=1)(hr_image[idx,:,:,:])
            # if random.random() < 0.5:
            #     lr_image = np.swapaxes(lr_image, 0, 1)
            #     hr_image = np.swapaxes(hr_image, 0, 1)
        return lr_image, hr_image

    def __len__(self):
        
        if self.mode == common.modes.TRAIN:
            return len(self.lr_files) * self.params.num_patches
        else:
            return len(self.lr_files)

class VideoSuperResolutionHdf5Dataset(VideoSuperResolutionDataset):

    def __init__(
            self,
            mode,
            params,
            lr_files,
            hr_files,
            lr_cache_file,
            hr_cache_file,
            lib_hdf5='h5py',
    ):
        super(VideoSuperResolutionHdf5Dataset, self).__init__(
            mode,
            params,
            lr_files,
            hr_files,
        )
        self.lr_cache_file = common.io.Hdf5(lr_cache_file, lib_hdf5)
        self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)

        cache_dir = os.path.dirname(lr_cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if not os.path.exists(lr_cache_file):
        # if 1==1:
            for lr_files_batch in self.lr_files:
                for lr_file_path in lr_files_batch:
                    lr_image = np.asarray(Image.open(lr_file_path))
                    lr_image = np.ascontiguousarray(lr_image)          
                    self.lr_cache_file.add(lr_file_path, lr_image)
                    print(lr_file_path, lr_image.shape)
        if self.mode != common.modes.PREDICT:
            if not os.path.exists(hr_cache_file):
            #if 1==1:
                for hr_files_batch in self.hr_files:
                    for hr_file_path in hr_files_batch:
                        hr_image = np.asarray(Image.open(hr_file_path))
                        hr_image = np.ascontiguousarray(hr_image)          
                        self.hr_cache_file.add(hr_file_path, hr_image)
                        print(hr_file_path, hr_image.shape)

    def _load_item(self, index):


        lr_image_list=[]
        hr_image_list=[]
        
        for lr_path in self.lr_files[index]:
            lr_image = self.lr_cache_file.get(lr_path)
            lr_image_list.append(lr_image)
        for hr_path in self.hr_files[index]:
            hr_image = self.hr_cache_file.get(hr_path)
            hr_image_list.append(hr_image)

        # for (lr_path,hr_path) in zip(self.lr_files[index],self.hr_files[index]):
            
        #     lr_image = self.lr_cache_file.get(lr_path)
        #     hr_image = self.hr_cache_file.get(hr_path)
            

        #     lr_image_list.append(lr_image)
        #     hr_image_list.append(hr_image)

            

        return lr_image_list, hr_image_list

class VideoSuperResolutionWithMVHdf5Dataset(VideoSuperResolutionDataset):

    def __init__(
            self,
            mode,
            params,
            lr_files,
            hr_files,
            lr_cache_file,
            hr_cache_file,
            mv_cache_file,
            lib_hdf5='h5py',
    ):
        super(VideoSuperResolutionWithMVHdf5Dataset, self).__init__(
            mode,
            params,
            lr_files,
            hr_files,
        )
        self.lr_cache_file = common.io.Hdf5(lr_cache_file, lib_hdf5)
        self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)
        self.mv_cache_file = common.io.Hdf5(mv_cache_file, lib_hdf5)

        cache_dir = os.path.dirname(lr_cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if not os.path.exists(lr_cache_file):
        # if 1==1:
            for lr_files_batch in self.lr_files:
                for lr_file_path in lr_files_batch:
                    lr_image = np.asarray(Image.open(lr_file_path))
                    lr_image = np.ascontiguousarray(lr_image)          
                    self.lr_cache_file.add(lr_file_path, lr_image)
                    print(lr_file_path, lr_image.shape)
        if not os.path.exists(mv_cache_file):
        # if 1==1:
            for lr_files_batch in self.lr_files:
                for lr_file_path in lr_files_batch:
                    file , frame = os.path.split(lr_file_path)
                    frame = int(frame.split(".")[0])
                    mv_npy = np.load(os.path.join(file,'hex-me16-ref1/motion.npy'))


                    self.mv_cache_file.add(lr_file_path, mv_npy[frame,:,:,:])
                    print(lr_file_path, mv_npy.shape)
        if self.mode != common.modes.PREDICT:
            if not os.path.exists(hr_cache_file):
            #if 1==1:
                for hr_files_batch in self.hr_files:
                    for hr_file_path in hr_files_batch:
                        hr_image = np.asarray(Image.open(hr_file_path))
                        hr_image = np.ascontiguousarray(hr_image)          
                        self.hr_cache_file.add(hr_file_path, hr_image)
                        print(hr_file_path, hr_image.shape)

    def __getitem__(self, index):
        
        if self.mode == common.modes.PREDICT:
            lr_image = np.asarray(Image.open(self.lr_files[index][1]))
            lr_image = transforms.functional.to_tensor(lr_image)
            return lr_image, self.hr_files[index][0]

        if self.mode == common.modes.TRAIN:
            index = index // self.params.num_patches
        
        lr_image_list, hr_image_list, mv_list = self._load_item(index) #numpy list
        lr_image_list_ = []
        hr_image_list_ = []
        mv_list_ = []
        # lr_image, hr_image = self._sample_patch(lr_image, hr_image)
        
        p1 = random.random()
        p2 = random.random()

        x = random.randrange(
            self.params.ignored_boundary_size, lr_image_list[0].shape[0] -
                                            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        y = random.randrange(
            self.params.ignored_boundary_size, lr_image_list[0].shape[1] -
                                            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        for (idx,(lr_image,hr_image,mv)) in enumerate(zip(lr_image_list,hr_image_list, mv_list)):
            if self.mode == common.modes.TRAIN:
                if self.params.train_sample_patch:
                    # lr_image, hr_image = self._augment(lr_image, hr_image,p1,p2)
                    lr_image, hr_image,mv = self._sample_patch(lr_image,hr_image,mv,x,y)
            lr_image = np.ascontiguousarray(lr_image)
            hr_image = np.ascontiguousarray(hr_image)
            mv = np.ascontiguousarray(mv)
            
            lr_image = transforms.functional.to_tensor(lr_image)
            hr_image = transforms.functional.to_tensor(hr_image)
            mv = torch.from_numpy(mv).permute(2,0,1)
            
            lr_image_list_.append(lr_image)
            hr_image_list_.append(hr_image)
            mv_list_.append(mv)
        lr_image = torch.stack(lr_image_list_)
        hr_image = torch.stack(hr_image_list_)
        mv = torch.stack(mv_list_)
        if self.mode == common.modes.TRAIN:
            lr_image, hr_image, mv = self._augment(lr_image, hr_image, mv,p1,p2)
        # print(lr_image.shape,hr_image.shape)


        mv = mv.float()
        if self.mode == common.modes.TRAIN:
            return torch.cat((lr_image, mv),dim=1), hr_image
        elif self.mode == common.modes.EVAL:

            p=str.split(os.path.splitext(self.lr_files[index][0])[0],"/")
            save_path = p[-2]+p[-1]
            # save_path = os.path.splitext(self.lr_files[index][0])[-2]+'_'+os.path.splitext(self.lr_files[index][0])[-1]
            return save_path,  torch.cat((lr_image, mv),dim=1), hr_image

    def _load_item(self, index):


        lr_image_list = []
        hr_image_list = []
        mv_list = []
        
        for (lr_path,hr_path) in zip(self.lr_files[index],self.hr_files[index]):
            
            lr_image = self.lr_cache_file.get(lr_path)
            hr_image = self.hr_cache_file.get(hr_path)
            mv = self.mv_cache_file.get(lr_path)
            

            lr_image_list.append(lr_image)
            hr_image_list.append(hr_image)
            mv_list.append(mv)

            

        return lr_image_list, hr_image_list, mv_list

    def _sample_patch(self, lr_image, hr_image,mv,x,y):
        
        if self.mode == common.modes.TRAIN:
            # sample patch while training

            lr_image = lr_image[x:x + self.params.lr_patch_size, y:y +
                                                                   self.params.lr_patch_size,:]
            mv = mv[x:x + self.params.lr_patch_size, y:y +
                                                                   self.params.lr_patch_size,:]                                                      
            hr_image = hr_image[x *
                                self.params.scale:(x + self.params.lr_patch_size) *
                                                  self.params.scale, y *
                                                                     self.params.scale:(y + self.params.lr_patch_size) *
                                                                                       self.params.scale,:]
        # else:
        #     hr_image = hr_image[:lr_image.shape[2] *
        #                          self.params.scale, :lr_image.shape[3] *
        #                                              self.params.scale]
        return lr_image, hr_image, mv

    def _augment(self, lr_image, hr_image, mv,p1,p2):

        if self.mode == common.modes.TRAIN:
            
            for idx in range(lr_image.shape[0]):
            # augmentation while training
                if p1 < 0.5:
                    lr_image[idx,:,:,:] = transforms.RandomHorizontalFlip(p=1)(lr_image[idx,:,:,:])
                    hr_image[idx,:,:,:] = transforms.RandomHorizontalFlip(p=1)(hr_image[idx,:,:,:])
                    mv[idx,:,:,:] = transforms.RandomHorizontalFlip(p=1)(mv[idx,:,:,:])
                if p2 < 0.5:
                    lr_image[idx,:,:,:] = transforms.RandomVerticalFlip(p=1)(lr_image[idx,:,:,:])
                    hr_image[idx,:,:,:] = transforms.RandomVerticalFlip(p=1)(hr_image[idx,:,:,:])
                    mv[idx,:,:,:] = transforms.RandomVerticalFlip(p=1)(mv[idx,:,:,:])
            # if random.random() < 0.5:
            #     lr_image = np.swapaxes(lr_image, 0, 1)
            #     hr_image = np.swapaxes(hr_image, 0, 1)
        return lr_image, hr_image, mv


class NemoHdf5Dataset(VideoSuperResolutionDataset):

    def __init__(
            self,
            mode,
            params,
            lr_files,
            hr_files,
            lr_cache_file,
            hr_cache_file,
            lib_hdf5='h5py',
    ):
        super(NemoHdf5Dataset, self).__init__(
            mode,
            params,
            lr_files,
            hr_files,
        )
        self.lr_cache_file = common.io.Hdf5(lr_cache_file, lib_hdf5)
        self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)
        

        cache_dir = os.path.dirname(lr_cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if not os.path.exists(lr_cache_file):
        # if 1==1:
            lr_has_writen = []
            for lr_files_batch in self.lr_files:
                for lr_file_path in lr_files_batch:
                    if lr_file_path in lr_has_writen:
                        continue
                    lr_image = np.fromfile(lr_file_path, dtype='uint8')
                    
                    lr_image = lr_image.reshape(240, 426 ,3)    
                    
                    self.lr_cache_file.add(lr_file_path, lr_image)
                    lr_has_writen.append(lr_file_path)
                    print(lr_file_path, lr_image.shape)
        if self.mode != common.modes.PREDICT:
            if not os.path.exists(hr_cache_file):
            #if 1==1:
                hr_has_writen = []
                for hr_files_batch in self.hr_files:
                    for hr_file_path in hr_files_batch:
                        if hr_file_path in hr_has_writen:
                            continue
                        hr_image = np.fromfile(hr_file_path, dtype='uint8')
                        
                        hr_image = hr_image.reshape(1080, 1920 ,3)   
                        # hr_image = hr_image[:, :,[2, 1,0]]         
                        self.hr_cache_file.add(hr_file_path, hr_image)
                        hr_has_writen.append(hr_file_path)
                        print(hr_file_path, hr_image.shape)

    def _load_item(self, index):


        lr_image_list=[]
        hr_image_list=[]
        
        for (lr_path,hr_path) in zip(self.lr_files[index],self.hr_files[index]):
            
            lr_image = self.lr_cache_file.get(lr_path)
            hr_image = self.hr_cache_file.get(hr_path)


            lr_image_list.append(lr_image)
            hr_image_list.append(hr_image)

            

        return lr_image_list, hr_image_list








