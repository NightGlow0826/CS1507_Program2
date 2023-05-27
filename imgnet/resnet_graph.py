#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : resnet_graph.py
@Author  : Gan Yuyang
@Time    : 2023/5/26 19:53
"""

from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.models as models

def draw_construction(model):
    writer_cons = SummaryWriter('graph')
    input = torch.randn(1, 3, 64, 64)
    output = model(input)
    writer_cons.add_graph(model, input)
    writer_cons.close()


if __name__ == '__main__':
    model_1: models.ResNet = models.__dict__['resnet18'](num_classes=200)
    draw_construction(model_1)
