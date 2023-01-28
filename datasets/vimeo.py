import os
import random
import common.modes
import datasets._vsr

LOCAL_DIR = '/data/zhuz/reds'
TRAIN_LR_DIR = '/data/zhuz/vimeo90/vimeo_septuplet/sequences'
TRAIN_HR_DIR = '/data/zhuz/vimeo90/vimeo_septuplet/sequences'
EVAL_LR_DIR = '/data/zhuz/vimeo90/vimeo_super_resolution_test/low_resolution'
EVAL_HR_DIR = '/data/zhuz/vimeo90/vimeo_super_resolution_test/target'

train_list = '/data/zhuz/vimeo90/vimeo_septuplet/sep_trainlist.txt'
test_list = '/data/zhuz/vimeo90/vimeo_septuplet/sep_testlist.txt'

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
        return Vimeo(mode, params)


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


class REDS_(datasets._vsr.VideoSuperResolutionDataset):

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


class Vimeo(datasets._vsr.VideoSuperResolutionHdf5Dataset):

    def __init__(self, mode, params):

        lr_cache_file = '/data/zhuz/cache/vimeo_{}_lr_x{}.h5'.format(mode, params.scale)
        hr_cache_file = '/data/zhuz/cache/vimeo_{}_hr.h5'.format(mode)
        self.mode = mode
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


        lr_files = list_image_files(lr_dir,image_iter,'lr',mode)
        if mode == common.modes.PREDICT:
            hr_files = lr_files
        else:
            hr_files = list_image_files(hr_dir,image_iter,'hr',mode)

        super(Vimeo, self).__init__(
            mode,
            params,
            lr_files,
            hr_files,
            lr_cache_file,
            hr_cache_file,
        )

def list_image_files(d, image_batch=10, resolution='lr',mode=common.modes.EVAL):
    files = sorted(os.listdir(d))
    files_lists_ = []
    for image_files in files:

        image_dir = os.path.join(d, image_files)
        image_list = sorted(os.listdir(image_dir))

        for image_files_3 in image_list:

            image_sequence = os.listdir(os.path.join(d, image_files,image_files_3))
            if mode == common.modes.TRAIN:
                image_sequence = [img for img in image_sequence if resolution in img]
            image_sequence = sorted(image_sequence)
            image_sequence = [os.path.join(d, image_files,image_files_3,img) for img in image_sequence]

            files_lists_.append(image_sequence)
    print(len(files_lists_))
    return files_lists_
