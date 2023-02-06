import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()
base_path = os.path.join(cwd, 'tieredImagenet', 'train_images')
val_path = os.path.join(cwd, 'tieredImagenet', 'val_images')
novel_path = os.path.join(cwd, 'tieredImagenet', 'test_images')

base = {}
base['label_names'] = []
base['image_names'] = []
base['image_labels'] = []
with open('base_labels.txt') as file:
    base['label_names'] = file.readlines()
for i in tqdm(range(351)):
    names = os.listdir(os.path.join(base_path, 'C%04d' % i))
    for name in names:
        base['image_names'].append(os.path.join(base_path, 'C%04d' % i, name))
        base['image_labels'].append(i)

val = {}
val['label_names'] = []
val['image_names'] = []
val['image_labels'] = []
with open('val_labels.txt') as file:
    val['label_names'] = file.readlines()
for i in tqdm(range(97)):
    names = os.listdir(os.path.join(val_path, 'C%04d' % i))
    for name in names:
        val['image_names'].append(os.path.join(val_path, 'C%04d' % i, name))
        val['image_labels'].append(i + 351)

novel = {}
novel['label_names'] = []
novel['image_names'] = []
novel['image_labels'] = []
with open('novel_labels.txt') as file:
    novel['label_names'] = file.readlines()
for i in tqdm(range(160)):
    names = os.listdir(os.path.join(novel_path, 'C%04d' % i))
    for name in names:
        novel['image_names'].append(os.path.join(novel_path, 'C%04d' % i, name))
        novel['image_labels'].append(i + 448)

json.dump(base, open('base.json', 'w'))
json.dump(val, open('val.json', 'w'))
json.dump(novel, open('novel.json', 'w'))

data = json.load(open('base.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))

data = json.load(open('val.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))

data = json.load(open('novel.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))
