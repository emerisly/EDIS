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

from utils import compute_map, compute_mrr, compute_ndcg, compute_recall
from evaluate_retrieval import get_metric_values_with_rank, load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="")
    parser.add_argument("-s", "--split", type=str, default="val", choices=['val', 'test'])
    opt = parser.parse_args()

    split = opt.split

    with open(op.join(opt.dir, "features", f"{split}_gt_rel.json"), "r") as file:
        gt_rel = json.load(file)

    sim_matrix = []

    text_embeds = torch.load(op.join(opt.dir, "features", f"{split}_text_feat.pt")).to("cuda")
    
    if "restricted" in opt.dir:
        image_feat_files = sorted(glob.glob(op.join(opt.dir, "features", f"{split}_image_*")))
    else:
        image_feat_files = sorted(glob.glob(op.join(opt.dir, "features", "val_image_*")))
    image_feat_files = sorted(glob.glob(op.join(opt.dir, "features", f"{split}_image_*")))
    for fname in image_feat_files:
        image_embeds = torch.load(fname).to("cuda")
        sim_matrix.append((text_embeds @ image_embeds.t()))
    sim_matrix = torch.cat(sim_matrix, dim=1)
    assert len(sim_matrix) == len(gt_rel)

    gt_rank = []
    top20 = []
    for sims, gt_idx in zip(sim_matrix, gt_rel):
        _, sorted_idx = torch.sort(sims, descending=True)
        idx_rank = torch.argsort(sorted_idx)
        gt_rank.append(idx_rank.cpu().numpy()[gt_idx[0]])
        top20.append(sorted_idx[:20].cpu().numpy().tolist())
    
    eval_results = get_metric_values_with_rank(gt_rel, gt_rank)
    eval_results['metadata'] = (eval_results['metadata'][0], eval_results['metadata'][1], top20)

    with open(op.join(opt.dir, f"{split}_results.json"), 'w') as file:
        json.dump(eval_results, file)
    
    