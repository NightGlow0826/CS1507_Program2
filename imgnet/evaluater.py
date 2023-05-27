#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : evaluater.py
@Author  : Gan Yuyang
@Time    : 2023/5/26 13:41
"""
import os

import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm

model_1: models.ResNet = models.__dict__['resnet18'](num_classes=200)
checkpoint1 = torch.load(r'checkpoint1admasingle3060.pth.tar')

model_2: models.ResNet = models.__dict__['resnet18'](num_classes=200)
checkpoint2 = torch.load(r'checkpointsingle3060.pth.tar')

new_state_dict = {}
for k, v in checkpoint1['state_dict'].items():
    name = k.replace('module.', '')  # remove the 'module.' prefix
    new_state_dict[name] = v

new_state_dict2 = {}
for k, v in checkpoint2['state_dict'].items():
    name = k.replace('module.', '')  # remove the 'module.' prefix
    new_state_dict2[name] = v

model_1.load_state_dict(new_state_dict)
model_1.eval()
model_2.load_state_dict(new_state_dict2)
model_2.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    os.path.join('tiny-imagenet-200', 'val'),
    transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, pin_memory=True)

lst = []
for i, (images, target) in tqdm(enumerate(val_loader), total=10000):
    output1: torch.tensor = model_1(images)
    output2: torch.tensor = model_2(images)
    o1, o2 = output1.flatten().argmax().item(), output2.flatten().argmax().item()
    # print(o1, o2, target.item())

    # if o1 != o2:
    #     lst.append(i)

    max1 = output1.topk(5, 1, True, True).indices.flatten().tolist()
    max2 = output2.topk(5, 1, True, True).indices.flatten().tolist()
    a = set(max1) & set(max2)
    if not a:
        # print(max1, max2, target.item())
        lst.append(i)
    # if
    if len(lst) == 10:
        break
    # quit()
print(lst)
path = r'tiny-imagenet-200/val'

k = 0
for folder in os.listdir(path):
    if not os.path.isdir(os.path.join(path, folder)):
        continue

    for imgpath in os.listdir(os.path.join(path, folder)):
        if k in lst:
            print(imgpath)
        k += 1
