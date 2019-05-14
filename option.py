###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################

import os
import argparse
import torch

# path_g = os.path.join(model_path, "cityscapes_global.800_4.5.2019.lr5e5.pth")
# # path_g = os.path.join(model_path, "fpn_global.804_nonorm_3.17.2019.lr2e5" + ".pth")
# path_g2l = os.path.join(model_path, "fpn_global2local.508_deep.cat.1x_fmreg_ensemble.p3.0.15l2_3.19.2019.lr2e5.pth")
# path_l2g = os.path.join(model_path, "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5.pth")
class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=7, help='segmentation classes')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3], help='mode for training procedure. 1: train global branch only. 2: train local branch with fixed global branch. 3: train global branch with fixed local branch')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=6, help='batch size for using local image patches')
        parser.add_argument('--size_g', type=int, default=508, help='size (in pixel) for downsampled global image')
        parser.add_argument('--size_p', type=int, default=508, help='size (in pixel) for cropped local image')
        parser.add_argument('--path_g', type=str, default="", help='name for global model path')
        parser.add_argument('--path_g2l', type=str, default="", help='name for local from global model path')
        parser.add_argument('--path_l2g', type=str, default="", help='name for global from local model path')
        parser.add_argument('--lamb_fmreg', type=float, default=0.15, help='loss weight feature map regularization')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr
        if args.mode == 1 or args.mode == 3:
            args.num_epochs = 120
            args.lr = 5e-5
        else:
            args.num_epochs = 50
            args.lr = 2e-5
        return args
