import torch
import os
import json
from PIL import Image, ImageFile
from collections import Counter


class VOCReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, year, transform):
        super(VOCReturnIndexDataset, self).__init__()
        id_file = os.path.join(root, "VOCdevkit/VOC%s/ImageSets/Main"%year, "trainval.txt")
        with open(id_file, "r") as file:
            ids = file.readlines()
        for i in range(len(ids)):
            ids[i] = ids[i].strip()

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.root = root
        self.year = year
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = os.path.join(self.root, "VOCdevkit/VOC%s"%self.year, "JPEGImages", "%s.jpg"%id)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, id
