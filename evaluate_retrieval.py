import datetime
import pdb
import re
import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision.transforms.functional import InterpolationMode
import glob
import os
import os.path as op
import json
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import collections
import argparse

from data import create_loader
from data.edis_dataset import edis_retrieval_full
import utils

from models.blip_retrieval import blip_retrieval
from models.mblip_retrieval import mblip_retrieval
from tqdm import tqdm
from PIL import Image
import yaml

from utils import compute_map, compute_mrr, compute_ndcg, compute_recall, compute_ndcg_topk

LOCAL_K = 10


@torch.no_grad()
def evaluation(model, data_loader, device, config, gt_rel, output_dir, split, n_chunks=1, mblip=False):
    # test
    model.eval() 
    
    caption_feat = not mblip 
    
    # print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []

    for i in tqdm(range(0, num_text, text_bs), desc="Computing features for text"):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds,dim=0) # (Nt, D)
    torch.save(text_embeds.cpu(), op.join(output_dir, f"{split}_text_feat.pt"))
    
    collapsed_sims_matrix = []
    chunk_size = len(data_loader) // n_chunks + 1
    chunk_idx = 0
    image_embeds = []
    caption_embeds = []
    for i, (image, caption, img_id) in enumerate(tqdm(data_loader, desc="Computing features for images")): 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)

        if mblip:
            image_atts = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(device)
            
            caption_input = model.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
            caption_ids = caption_input.input_ids

            caption_out = model.text_encoder(caption_ids,
                                            attention_mask=caption_input.attention_mask,
                                            encoder_hidden_states=image_feat,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True)
            image_embed = F.normalize(model.text_proj(caption_out.last_hidden_state[:,0,:]))
        else:   
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)
            if caption_feat:
                caption_input = model.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
                caption_output = model.text_encoder(caption_input.input_ids, attention_mask=caption_input.attention_mask, mode='text')
                caption_embed = F.normalize(model.text_proj(caption_output.last_hidden_state[:, 0, :]))
                caption_embeds.append(caption_embed)
        
        image_embeds.append(image_embed)

        if (i+1) % chunk_size == 0 and (i+1) != len(data_loader): # process the last chunk outside
            assert chunk_size == len(image_embeds)
            image_embeds = torch.cat(image_embeds, dim=0)
            torch.save(image_embeds.cpu(), op.join(output_dir, f"{split}_image_feat_chunk_{chunk_idx}.pt"))
            if caption_feat:
                caption_embeds = torch.cat(caption_embeds, dim=0)
                torch.save(caption_embeds.cpu(), op.join(output_dir, f"{split}_headline_feat_chunk_{chunk_idx}.pt"))
                caption_embeds = []

            image_embeds = []    
            chunk_idx += 1

    image_embeds = torch.cat(image_embeds, dim=0) # (Ni, D)
    torch.save(image_embeds.cpu(), op.join(output_dir, f"{split}_image_feat_chunk_{chunk_idx}.pt"))
    if caption_feat:
        caption_embeds = torch.cat(caption_embeds, dim=0)
        torch.save(caption_embeds.cpu(), op.join(output_dir, f"{split}_headline_feat_chunk_{chunk_idx}.pt"))


def get_metric_values_with_rank(gt_rel, gt_rank, silent=False):
    eval_results = {}

    # construct pred list of list
    pred_index = []
    gt_rel3 = []
    gt_rel2 = []
    for gt, rank in zip(gt_rel, gt_rank):
        gt_rel3.append(gt[0][:gt[1]])
        gt_rel2.append(gt[0][gt[1]:])
        pred = [-1] * max(max(rank)+1, 10)
        for idx, r in zip(gt[0], rank):
            pred[r] = idx
        pred_index.append(pred)
    
    msg = ""
    for topk in [1, 5, 10]:
        recall_k = compute_recall(gt_rel3, pred_index, topk, silent=silent) * 100
        ndcg_k = compute_ndcg_topk(gt_rel3, gt_rel2, pred_index, topk, silent=silent) * 100
        eval_results[f'recall@{topk}'] = recall_k
        eval_results[f'ndcg@{topk}'] = ndcg_k
        msg += f"recall@{topk}:{recall_k:.2f}, "

    map = compute_map(gt_rel3, gt_rank, silent=silent) * 100
    ndcg = compute_ndcg(gt_rel3, gt_rel2, gt_rank, silent=silent) * 100
    eval_results['mAP'] = map
    eval_results['NDCG'] = ndcg
    msg += f"mAP:{map:.2f}, NDCG:{ndcg:.2f}"
    if not silent:
        print(msg)
    eval_results['metadata'] = (gt_rel3, gt_rel2, pred_index)
    return eval_results


def load_data(opt, config, transformation, split):
    eval_dataset = edis_retrieval_full(transformation, config['image_root'], config['ann_root'], split, setting=opt.image_bank)
    print(f"Loaded {split} {opt.image_bank} datasets")

    samplers = [None]
    eval_loader = create_loader([eval_dataset],samplers,
                                batch_size=[config['batch_size_test']]*1,
                                num_workers=[4],
                                is_trains=[False], 
                                collate_fns=[None])
    return eval_loader[0], eval_dataset
                                            

def eval_split(opt, config, model, device, transformation,  split, is_mblip):    
    eval_loaders, eval_datasets = load_data(opt, config, transformation, split)

    gt_rel = []
    for k, v in eval_datasets.txt2img.items():
        assert len(v) != 0
        try:
            gt_rel.append((v + eval_datasets.txt2img_secondary[k], len(v)))
        except:
            gt_rel.append((v, len(v)))
    with open(op.join(opt.output_dir, "features", f"{split}_gt_rel.json"), "w") as file:
            json.dump(gt_rel, file)

    n_chunks = 1 if opt.image_bank == 'restricted' else 10
    evaluation(model, eval_loaders, device, config, gt_rel, op.join(opt.output_dir, "features"), split=split, n_chunks=n_chunks, mblip=is_mblip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_evaluate.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output/evaluate_results")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--image_bank", type=str, default="full", choices=['full', 'restricted'])
    parser.add_argument("--cuda", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=['val', 'test'])
    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.output_dir, "features"), exist_ok=True)

    cuda = opt.cuda
    device = "cuda:"+cuda if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    config = yaml.load(open(opt.config, "r"), Loader=yaml.Loader)
    image_root = config['image_root']
    ann_root = config['ann_root']

    if opt.checkpoint is not None:
        config['pretrained'] = opt.checkpoint

    # load model
    is_mblip = False
    if 'mblip' in config['pretrained']:
        f_model = mblip_retrieval
        is_mblip = True
    else:
        f_model = blip_retrieval

    model = f_model(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
    model = model.to(device)
    model.eval()

    # image preprocessing
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
            transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])  

    for split in [opt.split]:
        eval_split(opt, config, model, device, transform_test, split, is_mblip=is_mblip)



    