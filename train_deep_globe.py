#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
from utils.lovasz_losses import lovasz_softmax
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate, collate_test

n_class = 7

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

data_path = "/ssd1/chenwy/deep_globe/data/"

# model_path = "/home/chenwy/deep_globe/saved_models/"
model_path = "/home/chenwy/deep_globe/FPN_based/github/saved_models/"
if not os.path.isdir(model_path): os.mkdir(model_path)

# log_path = "/home/chenwy/deep_globe/runs/"
log_path = "/home/chenwy/deep_globe/FPN_based/github/runs/"
# log_path = "/home/jiangziyu/chenwy/deep_globe/runs/"
if not os.path.isdir(log_path): os.mkdir(log_path)


task_name = "test"
# task_name = "fpn_global.508_9.30.2018.lr2e5"
# task_name = "fpn_global2local.804_deep.cat.1x_fmreg_ensemble.p3.0.15l2_up_3.19.2019.lr2e5"
# task_name = "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5"

print(task_name)
###################################

mode = 2 # 1: train global; 2: train local from global; 3: train global from local
evaluation = False
test = evaluation and False
print("mode:", mode, "evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = 1
ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train", "Sat")) if is_image_file(image_name)][:2]
# ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train_test", "Sat")) if is_image_file(image_name)]
ids_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali", "Sat")) if is_image_file(image_name)][:2]
ids_test = [image_name for image_name in os.listdir(os.path.join(data_path, "offical_crossvali", "Sat")) if is_image_file(image_name)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = DeepGlobe(os.path.join(data_path, "train"), ids_train, label=True, transform=True)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True)
dataset_val = DeepGlobe(os.path.join(data_path, "crossvali"), ids_val, label=True)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
dataset_test = DeepGlobe(os.path.join(data_path, "offical_crossvali"), ids_test, label=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=10, collate_fn=collate_test, shuffle=False, pin_memory=True)

##### sizes are (w, h) ##############################
size0 = (2448, 2448)
# make sure margin / 32 is over 1.5 AND size_g is divisible by 4
# size_p/n/batch: 208/15/50, 308/11/22, 508/6/8, 804/4/4
size_g = (508, 508) # resized global image
size_p = (508, 508)
n = 6
sub_batch_size = 2
###################################
print("creating models......")

path_g = os.path.join(model_path, "fpn_global.resize512_9.2.2018.2.global" + ".pth")
# path_g = os.path.join(model_path, "fpn_global.804_nonorm_3.17.2019.lr2e5" + ".pth")
path_g2l = os.path.join(model_path, "fpn_global2local.508_deep.cat.1x_fmreg_ensemble.p3.0.15l2_3.19.2019.lr2e5.pth")
path_l2g = os.path.join(model_path, "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5.pth")
model, global_fixed = create_model_load_weights(n_class, mode, evaluation, path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g)

start_epoch = 0
if start_epoch > 0:
    print("restart epoch:", start_epoch)
    model.load_state_dict(torch.load(os.path.join(model_path, task_name + ".pth")))
###################################
if mode == 1:
    num_epochs = 120
    learning_rate = 2e-5
else:
    num_epochs = 50
    learning_rate = 2e-5
lamb_fmreg = 0.15
decay_fmreg = 1

optimizer = get_optimizer(model, mode, learning_rate=2e-5)

scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

criterion1 = FocalLoss(gamma=6)
criterion2 = nn.CrossEntropyLoss(ignore_index=-1)
criterion3 = lovasz_softmax
criterion = lambda x,y: criterion1(x, y)
# criterion = lambda x,y: 0.5*criterion1(x, y) + 0.5*criterion3(x, y)
mse = nn.MSELoss()

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(criterion, optimizer, n_class, size0, size_g, size_p, n, sub_batch_size, mode, lamb_fmreg)
evaluator = Evaluator(n_class, size0, size_g, size_p, n, sub_batch_size, mode, test)

best_pred = 0.0
print("start training......")
for epoch in range(start_epoch, num_epochs):
    trainer.set_train(model)
    optimizer.zero_grad()
    for i_batch, sample_batched in enumerate(tqdm(dataloader_train)):
        if evaluation: break
        scheduler(optimizer, i_batch, epoch, best_pred)
        trainer.train(sample_batched, model, global_fixed)

    score_train, score_train_global, score_train_local = trainer.get_scores()
    trainer.reset_metrics()
    # torch.cuda.empty_cache()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            if test: loader = tqdm(dataloader_test)
            else: loader = tqdm(dataloader_val)

            for i_batch, sample_batched in enumerate(loader):
                predictions, predictions_global, predictions_local = evaluator.eval_test(sample_batched, model, global_fixed)
                images = sample_batched['image']
                if not test:
                    labels = sample_batched['label'] # PIL images

                if test:
                    if not os.path.isdir("./prediction/"): os.mkdir("./prediction/")
                    for i in range(len(images)):
                        if mode == 1:
                            transforms.functional.to_pil_image(classToRGB(predictions_global[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")
                        else:
                            transforms.functional.to_pil_image(classToRGB(predictions[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")

                if not evaluation and not test:
                    if i_batch * batch_size + len(images) > (epoch % len(ids_val)) and i_batch * batch_size <= (epoch % len(ids_val)):
                        writer.add_image('image', transforms.ToTensor()(images[(epoch % len(ids_val)) - i_batch * batch_size]), epoch)
                        if not test:
                            writer.add_image('mask', classToRGB(np.array(labels[(epoch % len(ids_val)) - i_batch * batch_size])) * 255., epoch)
                        if mode == 2 or mode == 3:
                            writer.add_image('prediction', classToRGB(predictions[(epoch % len(ids_val)) - i_batch * batch_size]) * 255., epoch)
                            writer.add_image('prediction_local', classToRGB(predictions_local[(epoch % len(ids_val)) - i_batch * batch_size]) * 255., epoch)
                        writer.add_image('prediction_global', classToRGB(predictions_global[(epoch % len(ids_val)) - i_batch * batch_size]) * 255., epoch)

            # torch.cuda.empty_cache()

            # if not (test or evaluation): torch.save(model.state_dict(), "./saved_models/" + task_name + ".epoch" + str(epoch) + ".pth")
            if not (test or evaluation): torch.save(model.state_dict(), "./saved_models/" + task_name + ".pth")

            if test: break
            else:
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                evaluator.reset_metrics()
                if mode == 1:
                    if np.mean(np.nan_to_num(score_val_global["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val_global["iou"][1:]))
                    # if np.mean(np.nan_to_num(score_val_global["iou"])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val_global["iou"]))
                else:
                    if np.mean(np.nan_to_num(score_val["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"][1:]))
                    # if np.mean(np.nan_to_num(score_val["iou"])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"]))
                log = ""
                log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"][1:])), np.mean(np.nan_to_num(score_val["iou"][1:]))) + "\n"
                log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"][1:])), np.mean(np.nan_to_num(score_val_local["iou"][1:]))) + "\n"
                log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"][1:])), np.mean(np.nan_to_num(score_val_global["iou"][1:]))) + "\n"
                # log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"])), np.mean(np.nan_to_num(score_val["iou"]))) + "\n"
                # log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"])), np.mean(np.nan_to_num(score_val_local["iou"]))) + "\n"
                # log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"])), np.mean(np.nan_to_num(score_val_global["iou"]))) + "\n"
                log = log + "train: " + str(score_train["iou"]) + "\n"
                log = log + "val:" + str(score_val["iou"]) + "\n"
                log = log + "Local train:" + str(score_train_local["iou"]) + "\n"
                log = log + "Local val:" + str(score_val_local["iou"]) + "\n"
                log = log + "Global train:" + str(score_train_global["iou"]) + "\n"
                log = log + "Global val:" + str(score_val_global["iou"]) + "\n"
                log += "================================\n"
                print(log)
                if evaluation: break

                f_log.write(log)
                f_log.flush()
                if mode == 1:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train_global["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val_global["iou"][1:]))}, epoch)
                    # writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train_global["iou"])), 'validation iou': np.mean(np.nan_to_num(score_val_global["iou"]))}, epoch)
                else:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"][1:]))}, epoch)
                    # writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"]))}, epoch)

if not evaluation: f_log.close()