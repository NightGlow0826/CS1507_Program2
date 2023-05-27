#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : preprocess.py
@Author  : Gan Yuyang
@Time    : 2023/5/21 20:46
"""
import os
import re
ecd = 'utf-8'
if not os.path.exists('wktext'):
    os.makedirs(r'wktext')

for path in os.listdir(r'data/wikitext-2'):
    if path == 'README': continue
    with open(os.path.join('data/wikitext-2', path), 'r', encoding=ecd) as f:
        content = f.read()
        content = re.sub(r'<unk>', '', content)
        content = re.sub(r'@.@', '', content)
        content = re.sub(r'=.*=', '', content)
        content = re.sub(r'[\n\r]', '', content)
        content = re.sub(r'  ', '', content)
        content = re.sub(r'\s{2, }', '', content)
        # content = re.sub(r'[,.?!:;"]', '', content)
        content = re.sub(r'\(.*?\)', '', content)
        content = re.sub(r'\xa3', '', content)

    with open(os.path.join('wktext', path), 'w', encoding=ecd) as f:
        f.write(content)


