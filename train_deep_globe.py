#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
from utils.lovasz_losses import lovasz_softmax
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate, collate_test
from option import Options

args = Options().parse()
n_class = args.n_class

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

data_path = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################

mode = args.mode # 1: train global; 2: train local from global; 3: train global from local
evaluation = args.evaluation
test = evaluation and False
print("mode:", mode, "evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train", "Sat")) if is_image_file(image_name)]
ids_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali", "Sat")) if is_image_file(image_name)]
ids_test = [image_name for image_name in os.listdir(os.path.join(data_path, "offical_crossvali", "Sat")) if is_image_file(image_name)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = DeepGlobe(os.path.join(data_path, "train"), ids_train, label=True, transform=True)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True)
dataset_val = DeepGlobe(os.path.join(data_path, "crossvali"), ids_val, label=True)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
dataset_test = DeepGlobe(os.path.join(data_path, "offical_crossvali"), ids_test, label=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=10, collate_fn=collate_test, shuffle=False, pin_memory=True)

##### sizes are (w, h) ##############################
# make sure margin / 32 is over 1.5 AND size_g is divisible by 4
size_g = (args.size_g, args.size_g) # resized global image
size_p = (args.size_p, args.size_p) # cropped local patch size
sub_batch_size = args.sub_batch_size # batch size for train local patches
###################################
print("creating models......")

path_g = os.path.join(model_path, args.path_g)
path_g2l = os.path.join(model_path, args.path_g2l)
path_l2g = os.path.join(model_path, args.path_l2g)
model, global_fixed = create_model_load_weights(n_class, mode, evaluation, path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g)

###################################
num_epochs = args.num_epochs
learning_rate = args.lr
lamb_fmreg = args.lamb_fmreg

optimizer = get_optimizer(model, mode, learning_rate=learning_rate)

scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

criterion1 = FocalLoss(gamma=3)
criterion2 = nn.CrossEntropyLoss()
criterion3 = lovasz_softmax
criterion = lambda x,y: criterion1(x, y)
# criterion = lambda x,y: 0.5*criterion1(x, y) + 0.5*criterion3(x, y)
mse = nn.MSELoss()

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(criterion, optimizer, n_class, size_g, size_p, sub_batch_size, mode, lamb_fmreg)
evaluator = Evaluator(n_class, size_g, size_p, sub_batch_size, mode, test)

best_pred = 0.0
print("start training......")
for epoch in range(num_epochs):
    trainer.set_train(model)
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train); train_loss = 0
    for i_batch, sample_batched in enumerate(tbar):
        if evaluation: break
        scheduler(optimizer, i_batch, epoch, best_pred)
        loss = trainer.train(sample_batched, model, global_fixed)
        train_loss += loss.item()
        score_train, score_train_global, score_train_local = trainer.get_scores()
        if mode == 1: tbar.set_description('Train loss: %.3f; global mIoU: %.3f' % (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train_global["iou"]))))
        else: tbar.set_description('Train loss: %.3f; agg mIoU: %.3f' % (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train["iou"]))))

    score_train, score_train_global, score_train_local = trainer.get_scores()
    trainer.reset_metrics()
    # torch.cuda.empty_cache()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            if test: tbar = tqdm(dataloader_test)
            else: tbar = tqdm(dataloader_val)

            for i_batch, sample_batched in enumerate(tbar):
                predictions, predictions_global, predictions_local = evaluator.eval_test(sample_batched, model, global_fixed)
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                # use [1:] since class0 is not considered in deep_globe metric
                if mode == 1: tbar.set_description('global mIoU: %.3f' % (np.mean(np.nan_to_num(score_val_global["iou"])[1:])))
                else: tbar.set_description('agg mIoU: %.3f' % (np.mean(np.nan_to_num(score_val["iou"])[1:])))
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
                    if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
                        writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
                        if not test:
                            writer.add_image('mask', classToRGB(np.array(labels[(epoch % len(dataloader_val)) - i_batch * batch_size])) * 255., epoch)
                        if mode == 2 or mode == 3:
                            writer.add_image('prediction', classToRGB(predictions[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                            writer.add_image('prediction_local', classToRGB(predictions_local[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                        writer.add_image('prediction_global', classToRGB(predictions_global[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)

            # torch.cuda.empty_cache()

            # if not (test or evaluation): torch.save(model.state_dict(), "./saved_models/" + task_name + ".epoch" + str(epoch) + ".pth")
            if not (test or evaluation): torch.save(model.state_dict(), "./saved_models/" + task_name + ".pth")

            if test: break
            else:
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                evaluator.reset_metrics()
                if mode == 1:
                    if np.mean(np.nan_to_num(score_val_global["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val_global["iou"][1:]))
                else:
                    if np.mean(np.nan_to_num(score_val["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"][1:]))
                log = ""
                log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"][1:])), np.mean(np.nan_to_num(score_val["iou"][1:]))) + "\n"
                log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"][1:])), np.mean(np.nan_to_num(score_val_local["iou"][1:]))) + "\n"
                log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"][1:])), np.mean(np.nan_to_num(score_val_global["iou"][1:]))) + "\n"
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
                else:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"][1:]))}, epoch)

if not evaluation: f_log.close()