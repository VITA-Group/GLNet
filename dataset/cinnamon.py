import os
import random

import cv2
import numpy as np
# from os.path import join
import torch.utils.data as data
# import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import ToTensor


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 255]
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 0]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 255]
    transform = ToTensor()
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap)


class Cinnamon(data.Dataset):
    """ Custom Cinnamon datasets class """

    def __init__(self, root, ids, label=False, transform=False):
        super(Cinnamon, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.classdict = {0: "background", 1: "text", 2: "table border"}
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((2448, 2448))

    def __getitem__(self, index):
        sample = {}
        sample['id'] = self.ids[index][:-8]
        image = Image.open(os.path.join(
            self.root, "images/" + self.ids[index]))
        sample['image'] = image
        # TODO: Maybe need to create mask from json label or something
        if self.label:
            label = Image.open(os.path.join(
                 self.root, 'labels/' + self.ids[index].replace('.jpg', '_mask.jpg')))
            sample['label'] = label
        if self.transform and self.label:
            image, label = self._transform(image, label)
            sample['image'] = image
            sample['label'] = label
        return sample


    def _transform(self, image, label):
        # if np.random.random() > 0.5:
        #     image = self.color_jitter(image)

        # if np.random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     label = transforms.functional.vflip(label)

        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     degree = 60 * np.random.random() - 30
        #     image = transforms.functional.rotate(image, degree)
        #     label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     ratio = np.random.random()
        #     h = int(2448 * (ratio + 2) / 3.)
        #     w = int(2448 * (ratio + 2) / 3.)
        #     i = int(np.floor(np.random.random() * (2448 - h)))
        #     j = int(np.floor(np.random.random() * (2448 - w)))
        #     image = self.resizer(transforms.functional.crop(image, i, j, h, w))
        #     label = self.resizer(transforms.functional.crop(label, i, j, h, w))
        
        return image, label


    def __len__(self):
        return len(self.ids)
