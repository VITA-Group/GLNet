import glob
import os
import shutil
from random import shuffle

# Move all image in DeepGlobe/land-train/land-train/ to /DeepGlobe_mod/train/

data_path = '/home/linus/2DOCR/data/DeepGlobe_mod/'
[os.makedirs(data_path, os.path.join(data_path, each)) for each in ['val', 'test']]

train_path = os.path.join(data_path, 'train')
train_img = glob.glob(os.path.join(train_path, '*.jpg'))

def move_file(img_list, set_type):
    for old_img in img_list:
        new_img = old_img.replace('/train/', '/{}/'.format(set_type))
        old_mask = old_img.replace('_sat.jpg', '_mask.png')
        new_mask = old_mask.replace('/train/', '/{}/'.format(set_type))
        shutil.move(old_img, new_img)
        shutil.move(old_mask, new_mask)

shuffle(train_img)
val_img = train_img[:120]
train_img = train_img[120:]
move_file(val_img, 'val')

shuffle(train_img)
test_img = train_img[:40]
train_img = train_img[40:]
move_file(test_img, 'test')

