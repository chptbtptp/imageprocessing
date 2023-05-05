import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import json
import random
from tqdm import tqdm
import time

# modify
# CAT2IDX = {'CW_sanjianbanfanliu_A':0,'CW_sanjianbanfanliu_B':1}
CAT2IDX = {'CW_feidongmai_A':0,'CW_feidongmai_B':1}

def image_loader(path,transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')

    if transform is not None:
        image = transform(image)

    return image

class heartTUDataset(data.Dataset):
    def __init__(self, split, num_labels, root, img_root, transform=None,  testing=False):
        self.root = root
        self.phase = split
        self.img_list = []
        self.get_anno()
        self.num_classes = num_labels
        self.img_root = img_root
        self.transform = transform
        self.testing = testing
        self.epoch = 1

    def get_anno(self):
        self.img_list = json.loads(open(os.path.join(self.root,'heartTU_annotation.json'), 'r').read())[self.phase]
        self.cat2idx = CAT2IDX

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        item = self.img_list[index]

        return self.get(item)

    def get(self, item):
        image_ID = item
        # print(image_ID)
        img_name = os.path.join(self.img_root, image_ID)
        image = image_loader(img_name, self.transform)

        # modify
        labels_index = self.cat2idx['_'.join(image_ID.split('_')[:3])]
        labels = np.zeros(self.num_classes, np.float32)
        labels[labels_index] = 1
#         return image,labels
        return image, labels


