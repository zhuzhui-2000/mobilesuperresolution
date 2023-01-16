import os

import common.modes
import datasets._vsr

LOCAL_DIR = '/data/zhuz/reds'
TRAIN_LR_DIR = '/data/jinxinqi/Dataset/SuperResolution/NEMO-Dataset/1/image/240p_512kbps_s0_d300.webm'
TRAIN_HR_DIR = '/data/zhuz/nemo/1'
EVAL_LR_DIR = '/data/jinxinqi/Dataset/SuperResolution/NEMO-Dataset/1/image/240p_512kbps_s0_d300.webm'
EVAL_HR_DIR = '/data/zhuz/nemo/1'


def update_argparser(parser):
    datasets._vsr.update_argparser(parser)
    parser.add_argument(
        '--input_dir', help='Directory of input files in predict mode.')
    parser.set_defaults(
        num_channels=3,
        num_patches=1000,
        train_batch_size=16,
        eval_batch_size=1,
    )


def get_dataset(mode, params):
    if mode == common.modes.PREDICT:
        return REDS_(mode, params)
    else:
        return REDS(mode, params)


# class DIV2K(datasets._isr.ImageSuperResolutionHdf5Dataset):

#     def __init__(self, mode, params):
#         lr_cache_file = 'data/cache/div2k_{}_lr_x{}.h5'.format(mode, params.scale)
#         hr_cache_file = 'data/cache/div2k_{}_hr.h5'.format(mode)

#         lr_dir = {
#             common.modes.TRAIN: TRAIN_LR_DIR(params.scale),
#             common.modes.EVAL: EVAL_LR_DIR(params.scale),
#         }[mode]
#         hr_dir = {
#             common.modes.TRAIN: TRAIN_HR_DIR,
#             common.modes.EVAL: EVAL_HR_DIR,
#         }[mode]

#         lr_files = list_image_files(lr_dir)
#         if mode == common.modes.PREDICT:
#             hr_files = lr_files
#         else:
#             hr_files = list_image_files(hr_dir)

#         super(DIV2K, self).__init__(
#             mode,
#             params,
#             lr_files,
#             hr_files,
#             lr_cache_file,
#             hr_cache_file,
#         )


class REDS_(datasets._vsr.NemoHdf5Dataset):

    def __init__(self, mode, params):

        lr_dir = {
            common.modes.TRAIN: TRAIN_LR_DIR,
            common.modes.EVAL: EVAL_LR_DIR,
            common.modes.PREDICT: params.input_dir,
        }[mode]
        hr_dir = {
            common.modes.TRAIN: TRAIN_HR_DIR,
            common.modes.EVAL: EVAL_HR_DIR,
            common.modes.PREDICT: '',
        }[mode]

        lr_files = list_image_files(lr_dir)
        if mode == common.modes.PREDICT:
            hr_files = lr_files
        else:
            hr_files = list_image_files(hr_dir)

        super(REDS_, self).__init__(
            mode,
            params,
            lr_files,
            hr_files,
        )


class REDS(datasets._vsr.NemoHdf5Dataset):

    def __init__(self, mode, params):

        lr_cache_file = 'data/cache/nemo_{}_lr_x{}.h5'.format('train', params.scale)
        hr_cache_file = 'data/cache/nemo_{}_hr.h5'.format('train')

        lr_dir = {
            common.modes.TRAIN: TRAIN_LR_DIR,
            common.modes.EVAL: EVAL_LR_DIR,
        }[mode]
        hr_dir = {
            common.modes.TRAIN: TRAIN_HR_DIR,
            common.modes.EVAL: EVAL_HR_DIR,
        }[mode]
        if mode == common.modes.TRAIN:
            image_iter = params.image_batch
        else:
            image_iter = params.val_image_batch


        lr_files = list_image_files(lr_dir,mode,image_iter)
        if mode == common.modes.PREDICT:
            hr_files = lr_files
        else:
            hr_files = list_image_files(hr_dir,mode,image_iter)

        super(REDS, self).__init__(
            mode,
            params,
            lr_files,
            hr_files,
            lr_cache_file,
            hr_cache_file,
        )

def list_image_files(d,mode, image_batch=10):
    files = sorted(os.listdir(d))
    files = [os.path.join(d,f) for f in files if "_" not in f]
    file_num = len(files)
    
    files_lists_ = []
    if mode == common.modes.TRAIN:
        for batch in range(0,file_num+1-image_batch):
            files_lists_.append(files[batch:batch+image_batch])
    else:
        for batch in range(0,file_num+1-image_batch,image_batch):
            files_lists_.append(files[batch:batch+image_batch])
    
    return files_lists_
