from collections import defaultdict
import os
import os.path as op
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
from tqdm import tqdm
import random


class edis_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename = 'EDIS_train.json'
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0

        # NOTE: only gt image
        for ann in self.annotation:
            img_id = ann['candidates'][0]['candidate_id'] # we now use ground truth only
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1  
        
        # NOTE: all score 3 images
        # for ann in self.annotation:
        #     for img_ann in ann['candidates']: # use all samples with score 3
        #         img_id = img_ann['candidate_id']
        #         if img_ann['score'] == 3 and img_id not in self.img_ids.keys():
        #             self.img_ids[img_id] = n
        #     n += 1

    def __len__(self) -> int:
        return len(self.annotation)
    
    def __getitem__(self, index: int):
        ann = self.annotation[index]
        
        # NOTE: only gt image
        img_ann = ann['candidates'][0]

        # NOTE: all score 3 images
        # img_candidates = [img_ann for img_ann in ann['candidates'] if img_ann['score']==3]
        # img_ann = random.choice(img_candidates)

        image_path = os.path.join(self.image_root, os.path.basename(img_ann['image']))
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = pre_caption(img_ann['headline'], self.max_words)
        query = self.prompt + pre_caption(ann['query'], self.max_words)

        return image, caption, query, self.img_ids[img_ann['candidate_id']]


class edis_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''        
        self.annotation = json.load(open(os.path.join(ann_root, f"EDIS_{split}.json"),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.caption = []
        self.txt2img = {}
        self.img2txt = defaultdict(list)

        # NOTE: only gt image
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(os.path.join(self.image_root, os.path.basename(ann['candidates'][0]['image'])))
            self.img2txt[img_id] = []
            self.text.append(pre_caption(ann['query'], max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

        # NOTE: all score 3 images
        # img_id = 0
        # for txt_id, ann in enumerate(self.annotation):
        #     self.text.append(pre_caption(ann['query'], max_words))
        #     for img_ann in ann['candidates']:
        #         if op.basename(img_ann['image']) not in self.image:
        #             self.image.append(op.basename(img_ann['image']))
        #             self.caption.append(img_ann['headline'])
        #             if img_ann['score'] == 3:
        #                 self.img2txt[img_id].append(txt_id)
        #                 self.txt2img[txt_id] = img_id
        #             img_id += 1

    def __len__(self) -> int:
        return len(self.annotation)
    
    def __getitem__(self, index: int):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, os.path.basename(ann['candidates'][0]['image']))
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)  
        caption = pre_caption(ann['candidates'][0]['headline'], self.max_words)

        return image, caption, index


class edis_retrieval_full(edis_retrieval_eval):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, setting="full"):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''        
        self.annotation = json.load(open(os.path.join(ann_root, f"EDIS_{split}.json"),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.caption = []
        self.txt2img = defaultdict(list)
        self.img2txt = defaultdict(list)
        self.txt2img_secondary = defaultdict(list)
        self.image_ids = []

        if setting == 'restricted':
            img_id = 0
            for txt_id, ann in enumerate(tqdm(self.annotation, desc="loading text & images")):
                self.text.append(pre_caption(ann['query'], max_words))
                for img_ann in ann['candidates']:
                    self.image.append(os.path.basename(img_ann['image']))
                    self.caption.append(img_ann['headline'])
                    if img_ann['score'] == 3:
                        self.img2txt[img_id].append(txt_id)
                        self.txt2img[txt_id].append(img_id)
                    elif img_ann['score'] == 2:
                        self.txt2img_secondary[txt_id].append(img_id)
                    self.image_ids.append(img_ann['candidate_id'])
                    img_id += 1      
        elif setting == 'full':
            self.all_images = json.load(open(os.path.join(ann_root, "EDIS_candidates_1m.json"), "r"))
            img_id2idx = {d['id']:idx for idx, d in enumerate(self.all_images)}
            self.image_ids = [d['id'] for d in self.all_images]
            self.image = [op.basename(d['image']) for d in self.all_images]
            self.caption = [pre_caption(d['headline'], max_words) for d in self.all_images]

            for txt_id, ann in enumerate(tqdm(self.annotation, desc="loading text & images")):
                self.text.append(pre_caption(ann['query'], max_words))
                for img_ann in ann['candidates']:
                    img_id = img_id2idx[img_ann['candidate_id']]
                    if img_ann['score'] == 3:
                        self.img2txt[img_id].append(txt_id)
                        self.txt2img[txt_id].append(img_id)
                    elif img_ann['score'] == 2:
                        self.txt2img_secondary[txt_id].append(img_id)
        else:
            raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, index: int):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = self.caption[index]
        return image, caption, index