import torch
import numpy as np
from dataset.deep_globe import RGB_mapping_to_class, class_to_target

test = torch.randint(low=0, high=255, size=(6, 3, 127, 127), dtype=torch.uint8)

def rgb2mask(input):
    batch_size, num_channel, height, width = input.size()
    batch = []
    for i in range(batch_size):
        img = input[i, :, :, :].permute(1, 2, 0).numpy()
        classmap = RGB_mapping_to_class(img)
        batch.append(classmap)
    return torch.from_numpy(np.stack(batch, axis=0))

classmaps = rgb2mask(test)
target = class_to_target(classmaps)
import ipdb; ipdb.set_trace()
