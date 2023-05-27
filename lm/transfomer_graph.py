#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : transfomer_graph.py
@Author  : Gan Yuyang
@Time    : 2023/5/26 20:10
"""
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.models as models
import model

def draw_construction(model):
    writer_cons = SummaryWriter('graph')
    input = torch.randint(0, 100, (35, 20))
    output = model(input)
    writer_cons.add_graph(model, input)
    writer_cons.close()

if __name__ == '__main__':
    model = model.TransformerModel(78601, 200, 2, 200, 2, 0.2)
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)

    draw_construction(model)

