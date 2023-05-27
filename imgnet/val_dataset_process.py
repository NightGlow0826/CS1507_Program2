#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : val_dataset_process.py
@Author  : Gan Yuyang
@Time    : 2023/5/5 23:22
"""
import os
import shutil


# 因为我做分类，tinyimagenet一共有200个分类，val中每个分类有50张，所以先创建存放图片的文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path)


def move():
    # 现在根据val_annotations.txt，来提取出images中的每一类图片，根据每一行图片的信息，来指定对应文件夹存放
    with open("tiny-imagenet-200/val/val_annotations.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            dirlist = []
            imagelist = []
            dir = line.split()
            dir_name = dir[1:2]
            image_name = dir[0:1]
            dirlist.append(dir_name)
            imagelist.append(image_name)
            a = dirlist[0][0]
            b = imagelist[0][0]
            image_path = 'tiny-imagenet-200/val/images' + '/' + b
            dir_path = 'tiny-imagenet-200/val' + '/' + a
            shutil.copy(image_path, dir_path)


if __name__ == '__main__':
    # tinyimagenet提供了200分类的txt，winds.txt，根据它创建
    file = 'tiny-imagenet-200/val'
    with open('tiny-imagenet-200/wnids.txt', 'r') as w:
        for line in w.readlines():
            line = line.strip('\n')
            folder = file + '/' + line
            mkdir(folder)

    move()