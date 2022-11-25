import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *
from pathlib import Path


# urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
#         'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
#         'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

coco_pos_weight = [2.0934785783836416, 1.8372949068535294, 1.2327995986238534, 1.1366052548107424, 3.2814445970695973, 2.8689651721377105, 8.307583075734158, 2.564411378816794, 1.0927030392356172, 1.5113190636862084, 0.9948188506385343, 0.9991646528487779, 0.4349590329759256, 0.4416240448607346, 0.7487123903050564, 1.4709924055829229, 1.7712988136431045, 1.6921159294821346, 0.2450592130758885, 1.3690795338767194, 2.254616715604027, 1.6708132576935033, 0.27928639162401603, 1.6971917429744239, 1.860185585741478, 1.3195056462501535, 0.5205817191283293, 0.6841041428025965, 1.9517088779956429, 1.4974247806101129, 1.9499387810629423, 5.764081769436998, 1.9620391494798322, 4.008207494407159, 2.0951106022217894, 54.29299242424243, 0.8701645216124333, 1.6320043267041142, 3.684034441398218, 3.7653283712784593, 1.1844438629352139, 1.3835279922779924, 2.1629803822937625, 6.425590257023312, 1.2320931232091692, 4.7524370026525204, 1.6799519456165026, 3.2243588782243555, 8.365768482490273, 0.04095788962337836, 1.8467638721869095, 1.242488730929265, 4.0766069397042095, 1.8849750131509733, 2.4582694946261148, 7.25861748818366, 1.1305092543905775, 1.9162232620320856, 1.9393852606891577, 1.6175161751429432, 4.00372905027933, 1.7437165450121657, 1.6937155348983772, 5.421085476550681, 1.7361131298449615, 1.7548175808031343, 2.2428567702900066, 2.234000935162095, 1.6548664562807882, 47.777833333333334, 2.5860025258599952, 5.501541709314227, 0.8343691788264515, 2.3517857142857146, 1.0779116113506468, 1.8518540051679588, 0.9404262531712012, 1.6255878572508695, 1.3585255276127892, 2.027156798038846]

urls = {'train_img':'http://images.cocodataset.org/zips/train2017.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2017.zip',
        'test_img' : 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'}

def download_coco2017(root_, phase):
    cwd = os.getcwd()
    root = os.path.realpath(root_)
    print("root: ", root) #/home/seongha/LT-ML/data/coco
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, 'data/')
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2017.zip'
    elif phase == 'val':
        filename = 'val2017.zip'
    elif phase == 'test':
        filename = 'test2017.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(cwd)
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file,data)
        os.system(command)
    print('[dataset] Done!')

    # tmpdir = os.path.join(root, 'tmp/')
    # print(root, tmpdir)
    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2017.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls['annotations'], shell=True)
        os.chdir(cwd)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[annotation] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2017.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2017(data.Dataset):
    def __init__(self, root, transform=None, phase='train', mixup=False):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2017(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        self.mixup = mixup

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        # If data is for training, perform mixup, only perform mixup roughly on 1 for every 10 images
        if self.phase=="train" and self.mixup and index%10==0:
          return self.get_mixup(item)
        else: return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2017'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename), target

    def get_mixup(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2017'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1

        # Choose another image/label randomly
        mixup_idx = random.randint(0, len(self.img_list)-1)
        mixup_item = self.img_list[mixup_idx]
        filename = mixup_item['file_name']
        labels = sorted(mixup_item['labels'])
        mixup_img = Image.open(os.path.join(self.root, 'data', '{}2017'.format(self.phase), filename)).convert('RGB')
        mixup_target = np.zeros(self.num_classes, np.float32) - 1
        mixup_target[labels] = 1
        if self.transform:
            mixup_img = self.transform(mixup_img)

        # Select a random number from the given beta distribution
        # Mixup the images accordingly
        alpha = 0.1
        lam = np.random.beta(alpha, alpha)
        img = lam * img + (1 - lam) * mixup_img
        target = lam * target + (1 - lam) * mixup_target
        return (img, filename), target
