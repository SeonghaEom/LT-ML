#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
import os
import sys
sys.path.append('../')
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torchvision
from voc import *
from coco import *
import torchvision.transforms as transforms
from torchvision.models import resnet152, resnet101, resnet18, resnet34, resnet50
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
from config import seed_everything
seed_everything(0)


# coco

# In[3]:


path_csv = '../data/coco'

from collections import defaultdict, Counter
class_num = defaultdict(int)
with open(path_csv + '/data/train_anno.json') as f:
  adj = np.zeros((80,80))
  import json
  train = json.load(f)
  print(len(train), train[0])

  li = []
  gt_labels = np.zeros((len(train),80))
  img_id2idx = dict()
  idx2img_id = []
  for i,each in enumerate(train):
    li += each['labels']
    gt_labels[i, each['labels']] = 1
    for l in each['labels']:
      class_num[l] += 1

  nums = gt_labels.sum(axis=0)
  adj = []
  for i,col in enumerate(gt_labels.T):
    if i in [34]:
      print(i)
      selected = gt_labels[np.isin(col, [1.0]), :]
      nonzero_cnt = (selected != 0).sum(1)
      cnter = Counter(nonzero_cnt)
      print(cnter)
    cond_prob = gt_labels[np.isin(col,[1.0]),:].sum(axis=0)
    cond_prob[i] = 0
    adj.append(cond_prob)
    # print(adj[-1])
  nums = nums.tolist()
  nums.sort()
  nums.reverse()
  # nums = reversed(nums)
  print(max(nums), min(nums))
  di={'adj': np.asarray(adj), "nums": np.asarray(nums)}
  class_di = {k: v for k, v in sorted(class_num.items(), key=lambda item: item[1], reverse=True)} #sorted
print(class_di.keys(), class_di.values())
label_li = list(class_di.keys()) #coco


test_dataset = COCO2014('../data/coco', phase='val', inp_name='../data/coco/coco_glove_word2vec.pkl')
# train_dataset = Voc2007Classification('data/voc', 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl', LT=True)
# test_dataset = Voc2007Classification('data/voc', 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
# train_dataset = COCO2014('data/coco', phase='train', inp_name='data/coco/coco_glove_word2vec.pkl')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
test_dataset.transform = transforms.Compose([
                MultiScaleCrop(224, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.ToTensor(),
                normalize,
            ])

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)
from util import AveragePrecisionMeter
AP = AveragePrecisionMeter(difficult_examples=False)


# In[14]:



from models import *


m_li = [base_resnet50(80, pretrained=True), base_vit(80, image_size=224, pretrained=True), base_swin(80, image_size=224, pretrained=True)]
p_li = ['/home/seongha/LT-ML/checkpoint/coco/coco_LT(0)_label_count_resnet50-4-4-0_resnet50_base_best.pth.tar' ,  '/home/seongha/LT-ML/checkpoint/coco/coco_LT(0)_label_count_vit-4-4-0_vit_base_best.pth.tar',   '/home/seongha/LT-ML/checkpoint/coco/coco_LT(0)_label_count_swin-4-4-0_swin_base_best.pth.tar',   ]

def get_model(index):
  path = p_li[index]
  model = m_li[index]
  di = torch.load(path)
  print(di['best_score'])
  print(di.keys())
  model.load_state_dict(di['state_dict'])
  return model

def sort_by_class_size(ap_li):
  Sorted_Ap = []
  for k, v in class_di.items():
        idx = label_li.index(k)
        print("{}, {}, {}".format(k, v, ap_li[idx]))
        Sorted_Ap.append( ap_li[idx])

  return Sorted_Ap

name_map = {0: "resnet50", 1: "vit", 2: "swin"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(1).to(device)
model = model.eval()
for i, (input, target) in tqdm(enumerate(test_loader)):
  img, path, inp = input
  target[target == 0] = 1
  target[target == -1] = 0
  feat_Var = torch.autograd.Variable(img).float().to(device)
  
  # output = model(feat_Var, None).detach()
  output = model(feat_Var, None).detach()
  # print(output.requires_grad, target.requires_grad)
  AP.add(output, target)
map = 100 * AP.value().mean()
print(100 * AP.value())
ap_li = 100 * AP.value()


with open("AP_{}_fc.txt".format(name_map[1]), "w") as f:
  f.write (str(sort_by_class_size(ap_li)))

