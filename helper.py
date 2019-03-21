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
from models.fpn_global_local_fmreg_ensemble import fpn
from utils.metrics import ConfusionMatrix
from PIL import Image

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            # resized[i] = images[i].resize(shape, Image.NEAREST)
            resized[i] = images[i].resize(shape, Image.BILINEAR)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    # target -= 1 # make class 0 (should be ignored) as -1 (to be ignored in cross_entropy)
    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def global2patch(images, n, step, size):
    '''
    image/label => patches
    size: patch size
    return: list of PIL patch images
    '''
    patches = [ [images[i]] * (n ** 2) for i in range(len(images)) ]
    for i in range(len(images)):
        for x in range(n): # height / upper
            for y in range(n): # width / left
                patches[i][x * n + y] = transforms.functional.crop(images[i], x * step, y * step, size[0], size[1])
                # patches[i][x * n + y] = images[i].crop((y * step, x * step, y * step + size[1], x * step + size[0]))
    return patches

def patch2global(patches, n_class, n, step, size0, size_p, batch_size):
    '''
    merge softmax from predicted patches => whole complete predictions
    return: list of np.array
    '''
    predictions = [ np.zeros((n_class, size0[0], size0[1])) for i in range(batch_size) ]
    for i in range(batch_size):
        for j in range(n):
            for k in range(n):
                predictions[i][:, j * step: j * step + size_p[0], k * step: k * step + size_p[1]] += patches[i][j * n + k]
    return predictions

def template_patch2global(size0, size_p, n, step):
    template = np.zeros(size0)
    coordinates = [(0, 0)] * n ** 2
    patch = np.ones(size_p)
    step = (size0[0] - size_p[0]) // (n - 1)
    x = y = 0
    i = 0
    while x + size_p[0] <= size0[0]:
        while y + size_p[1] <= size0[1]:
            template[x:x+size_p[0], y:y+size_p[1]] += patch
            coordinates[i] = (1.0 * x / size0[0], 1.0 * y / size0[1])
            i += 1
            y += step
        x += step
        y = 0
    return Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda(), coordinates

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


def create_model_load_weights(n_class, mode=1, evaluation=False, path_g=None, path_g2l=None, path_l2g=None):
    model = fpn(n_class)
    model = nn.DataParallel(model)
    model = model.cuda()

    if (mode == 2 and not evaluation) or (mode == 1 and evaluation):
        # load fixed basic global branch
        partial = torch.load(path_g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if (mode == 3 and not evaluation) or (mode == 2 and evaluation):
        partial = torch.load(path_g2l)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    global_fixed = None
    if mode == 3:
        # load fixed basic global branch
        global_fixed = fpn(n_class)
        global_fixed = nn.DataParallel(global_fixed)
        global_fixed = global_fixed.cuda()
        partial = torch.load(path_g)
        state = global_fixed.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        global_fixed.load_state_dict(state)
        global_fixed.eval()

    if mode == 3 and evaluation:
        partial = torch.load(path_l2g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if mode == 1 or mode == 3:
        model.module.resnet_local.eval()
        model.module.fpn_local.eval()
    else:
        model.module.resnet_global.eval()
        model.module.fpn_global.eval()
    
    return model, global_fixed


def get_optimizer(model, mode=1, learning_rate=2e-5):
    if mode == 1 or mode == 3:
        # train global
        optimizer = torch.optim.Adam([
                {'params': model.module.resnet_global.parameters(), 'lr': learning_rate},
                {'params': model.module.resnet_local.parameters(), 'lr': 0},
                {'params': model.module.fpn_global.parameters(), 'lr': learning_rate},
                {'params': model.module.fpn_local.parameters(), 'lr': 0},
                {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    else:
        # train local
        optimizer = torch.optim.Adam([
                {'params': model.module.resnet_global.parameters(), 'lr': 0},
                {'params': model.module.resnet_local.parameters(), 'lr': learning_rate},
                {'params': model.module.fpn_global.parameters(), 'lr': 0},
                {'params': model.module.fpn_local.parameters(), 'lr': learning_rate},
                {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    return optimizer


class Trainer(object):
    def __init__(self, criterion, optimizer, n_class, size0, size_g, size_p, n, sub_batch_size=6, mode=1, lamb_fmreg=0.15):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size0 = size0
        self.size_g = size_g
        self.size_p = size_p
        self.n = n
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.lamb_fmreg = lamb_fmreg

        self.ratio = float(size_p[0]) / size0[0]
        self.step = (size0[0] - size_p[0]) // (n - 1)
        self.template, self.coordinates = template_patch2global(size0, size_p, n, self.step)
    
    def set_train(self, model):
        model.module.ensemble_conv.train()
        if self.mode == 1 or self.mode == 3:
            model.module.resnet_global.train()
            model.module.fpn_global.train()
        else:
            model.module.resnet_local.train()
            model.module.fpn_local.train()

    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def train(self, sample, model, global_fixed):
        images, labels = sample['image'], sample['label'] # PIL images
        labels_npy = masks_transform(labels, numpy=True) # label of origin size in numpy

        images_glb = resize(images, self.size_g) # list of resized PIL images
        images_glb = images_transform(images_glb)
        labels_glb = resize(labels, (self.size_g[0] // 4, self.size_g[1] // 4), label=True) # down 1/4 for loss
        # labels_glb = resize(labels, self.size_g, label=True)
        labels_glb = masks_transform(labels_glb)

        if self.mode == 2 or self.mode == 3:
            patches, label_patches = global2patch(images, self.n, self.step, self.size_p), global2patch(labels, self.n, self.step, self.size_p)
            predicted_patches = [ np.zeros((self.n**2, self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
            predicted_ensembles = [ np.zeros((self.n**2, self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
            outputs_global = [ None for i in range(len(images)) ]

        if self.mode == 1:
            # training with only (resized) global image #########################################
            outputs_global, _ = model.forward(images_glb, None, None, None)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            ##############################################

        if self.mode == 2:
            # training with patches ###########################################
            for i in range(len(images)):
                j = 0
                while j < self.n**2:
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], (self.size_p[0] // 4, self.size_p[1] // 4), label=True)) # down 1/4 for loss
                    # label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], self.size_p, label=True))

                    output_ensembles, output_global, output_patches, fmreg_l2 = model.forward(images_glb[i:i+1], patches_var, self.coordinates[j : j+self.sub_batch_size], self.ratio, mode=self.mode, n_patch=self.n**2) # include cordinates
                    loss = self.criterion(output_patches, label_patches_var) + self.criterion(output_ensembles, label_patches_var) + self.lamb_fmreg * fmreg_l2
                    loss.backward()

                    # patch predictions
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] = F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
                outputs_global[i] = output_global
            outputs_global = torch.cat(outputs_global, dim=0)

            self.optimizer.step()
            self.optimizer.zero_grad()
            #####################################################################################

        if self.mode == 3:
            # train global with help from patches ##################################################
            # go through local patches to collect feature maps
            # collect predictions from patches
            for i in range(len(images)):
                j = 0
                while j < self.n**2:
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    fm_patches, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, self.ratio, self.coordinates, [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=self.template, n_patch_all=self.n**2) # include cordinates
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
            # train on global image
            outputs_global, fm_global = model.forward(images_glb, None, self.coordinates, self.ratio, mode=self.mode, global_model=None, n_patch=self.n**2) # include cordinates
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward(retain_graph=True)
            # fmreg loss
            # generate ensembles & calc loss
            for i in range(len(images)):
                j = 0
                while j < self.n**2:
                    label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], (self.size_p[0] // 4, self.size_p[1] // 4), label=True))
                    # label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], self.size_p, label=True))
                    fl = fm_patches[i][j : j+self.sub_batch_size].cuda()
                    fg = model.module._crop_global(fm_global[i:i+1], self.coordinates[j:j+self.sub_batch_size], self.ratio)[0]
                    fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                    output_ensembles = model.module.ensemble(fl, fg)
                    # output_ensembles = F.interpolate(model.module.ensemble(fl, fg), self.size_p, **model.module._up_kwargs)
                    loss = self.criterion(output_ensembles, label_patches_var)# + 0.15 * mse(fl, fg)
                    if i == len(images) - 1 and j + self.sub_batch_size >= self.n**2:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)

                    # ensemble predictions
                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] = F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
            self.optimizer.step()
            self.optimizer.zero_grad()

        # global predictions ###########################
        predictions_global = F.interpolate(outputs_global.cpu(), self.size0).argmax(1).detach().numpy()
        self.metrics_global.update(labels_npy, predictions_global)

        if self.mode == 2 or self.mode == 3:
            # patch predictions ###########################
            scores_local = np.array(patch2global(predicted_patches, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))) # merge softmax scores from patches (overlaps)
            predictions_local = scores_local.argmax(1) # b, h, w
            self.metrics_local.update(labels_npy, predictions_local)
            ###################################################
            # combined/ensemble predictions ###########################
            scores = np.array(patch2global(predicted_ensembles, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))) # merge softmax scores from patches (overlaps)
            predictions = scores.argmax(1) # b, h, w
            self.metrics.update(labels_npy, predictions)


class Evaluator(object):
    def __init__(self, n_class, size0, size_g, size_p, n, sub_batch_size=6, mode=1, test=False):
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size0 = size0
        self.size_g = size_g
        self.size_p = size_p
        self.n = n
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.test = test

        self.ratio = float(size_p[0]) / size0[0]
        self.step = (size0[0] - size_p[0]) // (n - 1)
        self.template, self.coordinates = template_patch2global(size0, size_p, n, self.step)

        if test:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]
    
    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def eval_test(self, sample, model, global_fixed):
        with torch.no_grad():
            images = sample['image']
            if not self.test:
                labels = sample['label'] # PIL images
                labels_npy = masks_transform(labels, numpy=True)

            images_global = resize(images, self.size_g)
            outputs_global = np.zeros((len(images), self.n_class, self.size_g[0] // 4, self.size_g[1] // 4))
            # outputs_global = np.zeros((len(images), self.n_class, self.size_g[0], self.size_g[1]))
            if self.mode == 2 or self.mode == 3:
                images_local = [ image.copy() for image in images ]
                scores_local = np.zeros((len(images), self.n_class, self.size0[0], self.size0[1]))
                scores = np.zeros((len(images), self.n_class, self.size0[0], self.size0[1]))

            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                        images_global[b] = transforms.functional.hflip(images_global[b])
                        if self.mode == 2 or self.mode == 3:
                            images_local[b] = transforms.functional.rotate(images_local[b], 90) # rotate back!
                            images_local[b] = transforms.functional.hflip(images_local[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(images_global[b], 90)
                            if self.mode == 2 or self.mode == 3:
                                images_local[b] = transforms.functional.rotate(images_local[b], 90)

                    # prepare global images onto cuda
                    images_glb = images_transform(images_global) # b, c, h, w

                    if self.mode == 2 or self.mode == 3:
                        patches = global2patch(images_local, self.n, self.step, self.size_p)
                        predicted_patches = [ np.zeros((self.n**2, self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                        predicted_ensembles = [ np.zeros((self.n**2, self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]

                    if self.mode == 1:
                        # eval with only resized global image ##########################
                        if flip:
                            outputs_global += np.flip(np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global += np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2))
                        ################################################################

                    if self.mode == 2:
                        # eval with patches ###########################################
                        for i in range(len(images)):
                            j = 0
                            while j < self.n**2:
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                output_ensembles, output_global, output_patches, _ = model.forward(images_glb[i:i+1], patches_var, self.coordinates[j : j+self.sub_batch_size], self.ratio, mode=self.mode, n_patch=self.n**2) # include cordinates

                                # patch predictions
                                predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                                predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += patches_var.size()[0]
                            if flip:
                                outputs_global[i] += np.flip(np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1)), axis=2)
                            else:
                                outputs_global[i] += np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1))

                        if flip:
                            scores_local += np.flip(np.rot90(np.array(patch2global(predicted_patches, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                            scores += np.flip(np.rot90(np.array(patch2global(predicted_ensembles, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                        else:
                            scores_local += np.rot90(np.array(patch2global(predicted_patches, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                            scores += np.rot90(np.array(patch2global(predicted_ensembles, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                        ###############################################################

                    if self.mode == 3:
                        # eval global with help from patches ##################################################
                        # go through local patches to collect feature maps
                        # collect predictions from patches
                        for i in range(len(images)):
                            j = 0
                            while j < self.n**2:
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                fm_patches, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, self.ratio, self.coordinates, [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=self.template, n_patch_all=self.n**2) # include cordinates
                                predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += self.sub_batch_size
                        # go through global image
                        tmp, fm_global = model.forward(images_glb, None, self.coordinates, self.ratio, mode=self.mode, global_model=None, n_patch=self.n**2) # include cordinates
                        if flip:
                            outputs_global += np.flip(np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global += np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2))
                        # generate ensembles
                        for i in range(len(images)):
                            j = 0
                            while j < self.n**2:
                                fl = fm_patches[i][j : j+self.sub_batch_size].cuda()
                                fg = model.module._crop_global(fm_global[i:i+1], self.coordinates[j:j+self.sub_batch_size], self.ratio)[0]
                                fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                                output_ensembles = model.module.ensemble(fl, fg) # include cordinates
                                # output_ensembles = F.interpolate(model.module.ensemble(fl, fg), self.size_p, **model.module._up_kwargs)

                                # ensemble predictions
                                predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += self.sub_batch_size

                        if flip:
                            scores_local += np.flip(np.rot90(np.array(patch2global(predicted_patches, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                            scores += np.flip(np.rot90(np.array(patch2global(predicted_ensembles, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                        else:
                            scores_local += np.rot90(np.array(patch2global(predicted_patches, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                            scores += np.rot90(np.array(patch2global(predicted_ensembles, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                        ###################################################

            # global predictions ###########################
            predictions_global = F.interpolate(torch.Tensor(outputs_global), self.size0, mode='nearest').argmax(1).detach().numpy()
            if not self.test:
                self.metrics_global.update(labels_npy, predictions_global)

            if self.mode == 2 or self.mode == 3:
                # patch predictions ###########################
                predictions_local = scores_local.argmax(1) # b, h, w
                if not self.test:
                    self.metrics_local.update(labels_npy, predictions_local)
                ###################################################
                # combined/ensemble predictions ###########################
                predictions = scores.argmax(1) # b, h, w
                if not self.test:
                    self.metrics.update(labels_npy, predictions)
                return predictions, predictions_global, predictions_local
            else:
                return None, predictions_global, None
