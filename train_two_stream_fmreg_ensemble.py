#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from models.fpn_global_local_fmreg_ensemble_dynamic_size import fpn
from models.crf import dense_crf
from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
from utils.lovasz_losses import lovasz_softmax
from utils.metrics import ConfusionMatrix
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
import PIL
import time

n_class = 7

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

data_path = "/ssd1/chenwy/deep_globe/data/"

model_path = "/home/chenwy/deep_globe/saved_models/"
if not os.path.isdir(model_path): os.mkdir(model_path)

log_path = "/home/chenwy/deep_globe/runs/"
# log_path = "/home/jiangziyu/chenwy/deep_globe/runs/"
if not os.path.isdir(log_path): os.mkdir(log_path)

def resize(images, shape, label=False):
    ''' resize PIL images '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = transforms.functional.resize(images[i], shape, interpolation=PIL.Image.NEAREST)
        else:
            resized[i] = transforms.functional.resize(images[i], shape)
    return resized

def prepare_images(images):
    # PIL b, "w, h," c => cuda variable b, c, h, w
    shape = np.array(images[0]).shape
    if len(shape) == 3:
        h, w, c = shape
    else:
        h, w = shape
        c = 1
    if c == 1:
        images_var = torch.zeros((len(images), h, w)).type(torch.LongTensor)
    else:
        images_var = torch.zeros((len(images), c, h, w))
    for i in range(len(images)):
        if c == 1:
            images_var[i] = transforms.ToTensor()(images[i]).type(torch.LongTensor)[0, :, :]
        else:
            images_var[i] = transforms.ToTensor()(images[i])
            
    return Variable(images_var).cuda()

def restore_shape(predictions, images, pad=0):
    '''
    restore predictions' shape to the same size of origin labels
    predictions: numpy b, h, w
    images: origin PIL b, "w, h," c
    '''
    # restored = np.empty(len(predictions), dtype=object)
    restored = []
    for i in range(len(predictions)):
        (w, h) = images[i].size
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
        # restored[i] = np.round(cv2.resize(predictions[i] * 1.0, (w, h))) # h, w
        if pad == 0: restored.append(np.round(cv2.resize(predictions[i] * 1.0, (w, h), interpolation=cv2.INTER_NEAREST)).astype(np.int64)) # h, w
        else: restored.append(np.round(cv2.resize(predictions[i] * 1.0, (w, h), interpolation=cv2.INTER_NEAREST)).astype(np.int64)[pad:-pad, pad:-pad]) # h, w
    return restored

def resize_numpy(mat, h, w):
    ''' mat: c, h', w' '''
    c = mat.shape[0]
    resized = np.zeros((c, h, w))
    for i in range(c):
        resized[i] = cv2.resize(mat[i] * 1.0, (w, h), interpolation=cv2.INTER_NEAREST)
    return resized

def global2patch(images, n, step, size):
    '''
    image/label => patches
    size: patch size
    return: list of PIL patch images
    '''
    patches = [ [images[i]] * (n ** 2) for i in range(len(images)) ]
    for i in range(len(images)):
        for x in range(n):
            for y in range(n):
                patches[i][x * n + y] = transforms.functional.crop(images[i], x * step, y * step, size[0], size[1])
    return patches

def patch2global(patches, n, step, size0, size1, batch_size, template):
    '''
    predicted patches => predictions
    return: list of np.array
    '''
    predictions = [ np.zeros((n_class, size0[0], size0[1])) for i in range(batch_size) ]
    for i in range(batch_size):
        for j in range(n):
            for k in range(n):
                # print(predictions[i].shape, predictions[i][:, j * step: j * step + size1[0], k * step: k * step + size1[1]].shape, patches[i][j * n + k].shape)
                predictions[i][:, j * step: j * step + size1[0], k * step: k * step + size1[1]] += patches[i][j * n + k]
        # predictions[i] /= template
    return predictions

def template_patch2global(size0, size1, n, step):
    template = np.zeros(size0)
    coordinates = [(0, 0)] * n ** 2
    patch = np.ones(size1)
    step = (size0[0] - size1[0]) // (n - 1)
    x = y = 0
    i = 0
    while x + size1[0] <= size0[0]:
        while y + size1[1] <= size0[1]:
            template[x:x+size1[0], y:y+size1[1]] += patch
            coordinates[i] = (1.0 * x / size0[0], 1.0 * y / size0[1])
            i += 1
            y += step
        x += step
        y = 0
    return template, Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda(), coordinates

def one_hot_gaussian_blur(index, classes):
    '''
    index: numpy array b, h, w
    classes: int
    '''
    mask = np.transpose((np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2))
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    return {'image': image, 'label': label, 'id': id}

def collate_test(batch):
    image = [ b['image'] for b in batch ] # w, h
    id = [ b['id'] for b in batch ]
    return {'image': image, 'id': id}


task_name = "fpn_global.508_9.30.2018.lr2e5"
# task_name = "fpn_global2local.508.deep.cat.1x_ensemble_fmreg.p3_10.14.2018.lr2e5"
# task_name = "fpn_local2global.508_deep.cat_ensemble.p3_10.31.2018.lr2e5.local1x"

print(task_name)
###################################

mode = 1 # 1: train global; 2: train local from global; 3: train global from local
evaluation = False
test = evaluation and False
print("mode:", mode, "evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = 6
ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train", "Sat")) if is_image_file(image_name)]
# ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train_test", "Sat")) if is_image_file(image_name)]
ids_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali", "Sat")) if is_image_file(image_name)]
ids_test = [image_name for image_name in os.listdir(os.path.join(data_path, "offical_crossvali", "Sat")) if is_image_file(image_name)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = DeepGlobe(os.path.join(data_path, "train_test"), ids_train, label=True, transform=True)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=1, collate_fn=collate, shuffle=True, pin_memory=True)
dataset_val = DeepGlobe(os.path.join(data_path, "crossvali"), ids_val, label=True)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=1, collate_fn=collate, shuffle=False, pin_memory=True)
dataset_test = DeepGlobe(os.path.join(data_path, "offical_crossvali"), ids_test, label=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=1, collate_fn=collate_test, shuffle=False, pin_memory=True)

###################################
size0 = (2448, 2448)
# make sure margin / 32 is over 1.5 AND size1 is divisible by 4
size1 = (508, 508) # resized global image
# size_p/n/batch: 208/15/50, 308/11/22, 508/6/8, 804/4/4
size_p = (208, 208)
ratio = float(size_p[0]) / size0[0]
n = 6
step = (size0[0] - size_p[0]) // (n - 1)
sub_batch_size = 8

template_np, template, coordinates = template_patch2global(size0, size_p, n, step)
###################################
print("creating models......")

model = fpn(n_class)
model = nn.DataParallel(model)
model = model.cuda()

if (mode == 2 and not evaluation) or (mode == 1 and evaluation):
    # load fixed basic global branch
    model = fpn(n_class)
    model = nn.DataParallel(model)
    model = model.cuda()
    partial = torch.load(os.path.join(model_path, "fpn_global.resize512_9.2.2018.2.global" + ".pth"))
    state = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
    # 2. overwrite entries in the existing state dict
    state.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(state)

if (mode == 3 and not evaluation) or (mode == 2 and evaluation):
    partial = torch.load(os.path.join(model_path, "fpn_global2local.508.deep.cat.1x_ensemble_fmreg.p3_10.14.2018.lr2e5" + ".pth"))
    state = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
    # 2. overwrite entries in the existing state dict
    state.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(state)

if mode == 3:
    # load fixed basic global branch
    global_fixed = fpn(n_class)
    global_fixed = nn.DataParallel(global_fixed)
    global_fixed = global_fixed.cuda()
    partial = torch.load(os.path.join(model_path, "fpn_global.resize512_9.2.2018.2.global" + ".pth"))
    state = global_fixed.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
    # 2. overwrite entries in the existing state dict
    state.update(pretrained_dict)
    # 3. load the new state dict
    global_fixed.load_state_dict(state)
    global_fixed.eval()

if mode == 3 and evaluation:
    # partial = torch.load(os.path.join(model_path, "fpn_local2global.508_deep.cat_ensemble.p3_10.15.2018.lr2e5" + ".pth"))
    partial = torch.load(os.path.join(model_path, "fpn_local2global.508_deep.cat_ensemble.p3_10.31.2018.lr2e5.local1x" + ".pth"))
    state = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
    # 2. overwrite entries in the existing state dict
    state.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(state)

start_epoch = 0
if start_epoch > 0:
    print("restart epoch:", start_epoch)
    model.load_state_dict(torch.load(os.path.join(model_path, task_name + ".pth")))
###################################
if mode == 1: num_epochs = 200
else: num_epochs = 100
learning_rate = 2e-5
lamb_fmreg = 0.15
decay_fmreg = 1

if mode == 1 or mode == 3:
    # train global
    optimizer = torch.optim.Adam([
            {'params': model.module.resnet_global.parameters(), 'lr': learning_rate},
            {'params': model.module.resnet_local.parameters(), 'lr': 0},
            {'params': model.module.fpn_global.parameters(), 'lr': learning_rate},
            {'params': model.module.fpn_local.parameters(), 'lr': 0},
            {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
        ], weight_decay=5e-4)
    model.module.resnet_local.eval()
    model.module.fpn_local.eval()
else:
    # train local
    optimizer = torch.optim.Adam([
            {'params': model.module.resnet_global.parameters(), 'lr': 0},
            {'params': model.module.resnet_local.parameters(), 'lr': learning_rate},
            {'params': model.module.fpn_global.parameters(), 'lr': 0},
            {'params': model.module.fpn_local.parameters(), 'lr': learning_rate},
            {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
        ], weight_decay=5e-4)
    model.module.resnet_global.eval()
    model.module.fpn_global.eval()

scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

# criterion = FocalLoss(device, gamma=6)#, one_hot=False)
criterion = nn.CrossEntropyLoss(ignore_index=0)
mse = nn.MSELoss()
# criterion.to(device)

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

metrics = ConfusionMatrix(n_class)
metrics_local = ConfusionMatrix(n_class)
metrics_global = ConfusionMatrix(n_class)

best_pred = 0.0
print("start training......")
for epoch in range(start_epoch, num_epochs):
    model.module.ensemble_conv.train()
    if mode == 1 or mode == 3:
        model.module.resnet_global.train()
        model.module.fpn_global.train()
    else:
        model.module.resnet_local.train()
        model.module.fpn_local.train()
    optimizer.zero_grad()
    for i_batch, sample_batched in enumerate(tqdm(dataloader_train)):
        if evaluation: break

        scheduler(optimizer, i_batch, epoch, best_pred)

        images, labels = sample_batched['image'], sample_batched['label'] # PIL images
        images_var, labels_var = resize(images, size1), resize(labels, (size1[0] // 4, size1[1] // 4), label=True) # list of resized PIL images
        images_var = prepare_images(images_var)
        labels_var = prepare_images(labels_var)

        if mode == 2 or mode == 3:
            patches, label_patches = global2patch(images, n, step, size_p), global2patch(labels, n, step, size_p)
            predicted_patches = [ np.zeros((n**2, n_class, size_p[0], size_p[1])) for i in range(len(images)) ]
            predicted_ensembles = [ np.zeros((n**2, n_class, size_p[0], size_p[1])) for i in range(len(images)) ]
            outputs_global = [ None for i in range(len(images)) ]

        if mode == 1:
            # training with only (resized) global image #########################################
            outputs_global, output_patches = model.forward(images_var, None, None, None)
            loss = criterion(outputs_global, labels_var)
            # loss = lovasz_softmax(outputs_global, labels_var)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ##############################################

        if mode == 2:
            # training with patches ###########################################
            for i in range(len(images)):
                j = 0
                while j < n * n:
                    patches_var = prepare_images(patches[i][j : j+sub_batch_size]) # b, c, h, w
                    label_patches_var = resize(label_patches[i][j : j+sub_batch_size], (size_p[0] // 4, size_p[1] // 4)) # b, c, h, w
                    label_patches_var = prepare_images(label_patches_var)

                    output_ensembles, output_global, output_patches, fmreg_l2 = model.forward(images_var[i:i+1], patches_var, coordinates[j : j+sub_batch_size], ratio, mode=mode, n_patch=n*n) # include cordinates
                    loss = criterion(output_patches, label_patches_var) + criterion(output_ensembles, label_patches_var) + (lamb_fmreg * decay_fmreg ** epoch) * fmreg_l2
                    loss.backward()
                    
                    # patch predictions
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.upsample(output_patches, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] = F.upsample(output_ensembles, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                    j += sub_batch_size
                outputs_global[i] = output_global
            outputs_global = torch.cat(outputs_global, dim=0)

            optimizer.step()
            optimizer.zero_grad()
            #####################################################################################

        if mode == 3:
            # train global with help from patches ##################################################
            # go through local patches to collect feature maps
            # collect predictions from patches
            for i in range(len(images)):
                j = 0
                while j < n * n:
                    patches_var = prepare_images(patches[i][j : j+sub_batch_size]) # b, c, h, w
                    fm_patches, output_patches = model.module.collect_local_fm(images_var[i:i+1], patches_var, ratio, coordinates, [j, j+sub_batch_size], len(images), global_model=global_fixed, template=template, n_patch_all=n*n) # include cordinates
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.upsample(output_patches, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                    j += sub_batch_size
            # train on global image
            outputs_global, fm_global = model.forward(images_var, None, coordinates, ratio, mode=mode, global_model=None, n_patch=n*n) # include cordinates
            loss = criterion(outputs_global, labels_var)
            loss.backward(retain_graph=True)
            # fmreg loss
            # generate ensembles & calc loss
            for i in range(len(images)):
                j = 0
                while j < n * n:
                    label_patches_var = prepare_images(resize(label_patches[i][j : j+sub_batch_size], (size_p[0] // 4, size_p[1] // 4))) # b, c, h, w
                    fl = fm_patches[i][j : j+sub_batch_size].cuda()
                    fg = model.module._crop_global(fm_global[i:i+1], coordinates[j:j+sub_batch_size], ratio)[0]
                    fg = F.upsample(fg, size=fl.size()[2:], mode='bilinear')
                    output_ensembles = model.module.ensemble(fl, fg) # include cordinates
                    loss = criterion(output_ensembles, label_patches_var)# + 0.15 * mse(fl, fg)
                    if i == len(images) - 1 and j + sub_batch_size >= n * n:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)
                    
                    # ensemble predictions
                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] = F.upsample(output_ensembles, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                    j += sub_batch_size
            optimizer.step()
            optimizer.zero_grad()

        if mode == 2 or mode == 3:
            # patch predictions ###########################
            scores_local = np.array(patch2global(predicted_patches, n, step, size0, size_p, len(images), template_np)) # merge softmax scores from patches (overlaps)
            predictions_local = scores_local.argmax(1) # b, h, w
            metrics_local.update([ np.array(m).astype(np.int64) for m in labels ], predictions_local)
            ###################################################
        
        # global predictions ###########################
        outputs_global = outputs_global.data.cpu().numpy()
        predictions_global = restore_shape(outputs_global.argmax(1), images) # b, h, w
        metrics_global.update([ np.array(m).astype(np.int64) for m in labels ], predictions_global)

        if mode == 2 or mode == 3:
            # combined/ensemble predictions ###########################
            scores = np.array(patch2global(predicted_ensembles, n, step, size0, size_p, len(images), template_np)) # merge softmax scores from patches (overlaps)
            predictions = scores.argmax(1) # b, h, w
            metrics.update([ np.array(m).astype(np.int64) for m in labels ], predictions)
    score_train = metrics.get_scores()
    score_train_local = metrics_local.get_scores()
    score_train_global = metrics_global.get_scores()
    metrics.reset()
    metrics_local.reset()
    metrics_global.reset()
    # torch.cuda.empty_cache()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            # if epoch > 0 and epoch % 2 == 1:
            if test:
                loader = tqdm(dataloader_test)
                flip_range = [False, True]
                rotate_range = [0, 1, 2, 3]
            else:
                loader = tqdm(dataloader_val)
                flip_range = [False]
                rotate_range = [0]

            for i_batch, sample_batched in enumerate(loader):
                pad = 0

                if pad > 0: images = [transforms.functional.pad(image, pad, padding_mode='symmetric') for image in sample_batched['image']] # PIL images
                else: images = sample_batched['image']
                if not test:
                    labels = sample_batched['label'] # PIL images

                images_global = resize(images, size1)
                outputs_global = np.zeros((len(images), n_class, size1[0] // 4, size1[1] // 4))
                if mode == 2 or mode == 3:
                    images_local = [ image.copy() for image in images ]
                    scores_local = np.zeros((len(images), n_class, size0[0], size0[1]))
                    scores = np.zeros((len(images), n_class, size0[0], size0[1]))

                for flip in flip_range:
                    if flip:
                        # we already rotated images for 270'
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                            images_global[b] = transforms.functional.hflip(images_global[b])
                            if mode == 2 or mode == 3:
                                images_local[b] = transforms.functional.rotate(images_local[b], 90) # rotate back!
                                images_local[b] = transforms.functional.hflip(images_local[b])
                    for angle in rotate_range:
                        if angle > 0:
                            for b in range(len(images)):
                                images_global[b] = transforms.functional.rotate(images_global[b], 90)
                                if mode == 2 or mode == 3:
                                    images_local[b] = transforms.functional.rotate(images_local[b], 90)

                        # prepare global images onto cuda
                        images_var = prepare_images(images_global) # b, c, h, w

                        if mode == 2 or mode == 3:
                            patches = global2patch(images_local, n, step, size_p)
                            predicted_patches = [ np.zeros((n**2, n_class, size_p[0], size_p[1])) for i in range(len(images)) ]
                            predicted_ensembles = [ np.zeros((n**2, n_class, size_p[0], size_p[1])) for i in range(len(images)) ]

                        if mode == 1:
                            # eval with only resized global image ##########################
                            if flip:
                                outputs_global += np.flip(np.rot90(model.forward(images_var, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                            else:
                                outputs_global += np.rot90(model.forward(images_var, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2))
                            ################################################################

                        if mode == 2:
                            # eval with patches ###########################################
                            for i in range(len(images)):
                                j = 0
                                while j < n * n:
                                    patches_var = prepare_images(patches[i][j : j+sub_batch_size]) # b, c, h, w
                                    output_ensembles, output_global, output_patches, _ = model.forward(images_var[i:i+1], patches_var, coordinates[j : j+sub_batch_size], ratio, mode=mode, n_patch=n*n) # include cordinates
                                    
                                    # patch predictions
                                    predicted_patches[i][j:j+output_patches.size()[0]] += F.upsample(output_patches, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.upsample(output_ensembles, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                                    j += patches_var.size()[0]
                                if flip:
                                    outputs_global[i] += np.flip(np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1)), axis=2)
                                else:
                                    outputs_global[i] += np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1))

                            if flip:
                                scores_local += np.flip(np.rot90(np.array(patch2global(predicted_patches, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                scores += np.flip(np.rot90(np.array(patch2global(predicted_ensembles, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                            else:
                                scores_local += np.rot90(np.array(patch2global(predicted_patches, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores += np.rot90(np.array(patch2global(predicted_ensembles, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                            ###############################################################

                        if mode == 3:
                            # eval global with help from patches ##################################################
                            # go through local patches to collect feature maps
                            # collect predictions from patches
                            for i in range(len(images)):
                                j = 0
                                while j < n * n:
                                    patches_var = prepare_images(patches[i][j : j+sub_batch_size]) # b, c, h, w
                                    fm_patches, output_patches = model.module.collect_local_fm(images_var[i:i+1], patches_var, ratio, coordinates, [j, j+sub_batch_size], len(images), global_model=global_fixed, template=template, n_patch_all=n*n) # include cordinates
                                    predicted_patches[i][j:j+output_patches.size()[0]] += F.upsample(output_patches, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                                    j += sub_batch_size
                            # go through global image
                            tmp, fm_global = model.forward(images_var, None, coordinates, ratio, mode=mode, global_model=None, n_patch=n*n) # include cordinates
                            if flip:
                                outputs_global += np.flip(np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                            else:
                                outputs_global += np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2))
                            # generate ensembles
                            for i in range(len(images)):
                                j = 0
                                while j < n * n:
                                    fl = fm_patches[i][j : j+sub_batch_size].cuda()
                                    fg = model.module._crop_global(fm_global[i:i+1], coordinates[j:j+sub_batch_size], ratio)[0]
                                    fg = F.upsample(fg, size=fl.size()[2:], mode='bilinear')
                                    output_ensembles = model.module.ensemble(fl, fg) # include cordinates
                                    
                                    # ensemble predictions
                                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.upsample(output_ensembles, size=(size_p[0], size_p[1]), mode='nearest').data.cpu().numpy()
                                    j += sub_batch_size

                            if flip:
                                scores_local += np.flip(np.rot90(np.array(patch2global(predicted_patches, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                scores += np.flip(np.rot90(np.array(patch2global(predicted_ensembles, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                            else:
                                scores_local += np.rot90(np.array(patch2global(predicted_patches, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores += np.rot90(np.array(patch2global(predicted_ensembles, n, step, size0, size_p, len(images), template_np)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                            ###################################################
        
                if mode == 2 or mode == 3:
                    # patch predictions ###########################
                    predictions_local = scores_local.argmax(1) # b, h, w
                    if not test:
                        metrics_local.update([ np.array(m).astype(np.int64) for m in labels ], predictions_local)
                    ###################################################
                
                # global predictions ###########################
                predictions_global = restore_shape(outputs_global.argmax(1), images, pad=pad) # b, h, w
                if not test:
                    metrics_global.update([ np.array(m).astype(np.int64) for m in labels ], predictions_global)

                if mode == 2 or mode == 3:
                    # combined/ensemble predictions ###########################
                    predictions = scores.argmax(1) # b, h, w
                    if not test:
                        metrics.update([ np.array(m).astype(np.int64) for m in labels ], predictions)

                if test:
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

            if not (test or evaluation): torch.save(model.state_dict(), "./saved_models/" + task_name + ".epoch" + str(epoch) + ".pth")
            # if not (test or evaluation): torch.save(model.state_dict(), "./saved_models/" + task_name + ".pth")

            if test: break
            else:
                score_val = metrics.get_scores()
                score_val_local = metrics_local.get_scores()
                score_val_global = metrics_global.get_scores()
                metrics.reset()
                metrics_local.reset()
                metrics_global.reset()
                if mode == 1:
                    if np.mean(np.nan_to_num(score_val_global["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val_global["iou"][1:]))
                else:
                    if np.mean(np.nan_to_num(score_val["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"][1:]))
                log = ""
                # log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train["iou_mean"], score_val["iou_mean"]) + "\n"
                # log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train_local["iou_mean"], score_val_local["iou_mean"]) + "\n"
                # log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train_global["iou_mean"], score_val_global["iou_mean"]) + "\n"
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
                # print('epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train["iou_mean"], score_val["iou_mean"])) 
                # print('epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train_local["iou_mean"], score_val_local["iou_mean"])) 
                # print('epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train_global["iou_mean"], score_val_global["iou_mean"])) 
                # print("train:", score_train["iou"])
                # print("val:", score_val["iou"])
                # print("Local train:", score_train_local["iou"])
                # print("Local val:", score_val_local["iou"])
                # print("Global train:", score_train_global["iou"])
                # print("Global val:", score_val_global["iou"])
                if mode == 1:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train_global["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val_global["iou"][1:]))}, epoch)
                else:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"][1:]))}, epoch)

f_log.close()