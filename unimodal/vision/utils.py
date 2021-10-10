
import json, time, os, base64
import numpy as np
from collections import Counter
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from model import Model
from dataset import ImageDataset
from easydict import EasyDict as edict

def load_json(cfg_path):
    """
    Function to load config file
        Args:
            cfg_path      : path of config file
        Returns:
            cfg           : easydict config object for
                            cfg.json file
    """
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    return cfg

def load_data(args, cfg):
    trainval_path =  cfg.trainval_path
    dataset = json.load(open(trainval_path, "r"))
    if args.verbose:
        print("Data Split: ", Counter([dataset[k]['split'] for k in dataset]))
        print("Total queries: ", len(set([dataset[k]['Guid'] for k in dataset])))
        print("Query Types: ", Counter([dataset[k]['Qcate'] for k in dataset]))
    return dataset

def get_ids(args, cfg, dataset):
    query_type = cfg.Qcate
    ids = [id for id in dataset if dataset[id]["Qcate"] == query_type]
    if args.verbose:
        print(f"Total ids for query {query_type}: ", len(ids))
    return ids

def get_model(args, cfg):
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    model = Model(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    if args.verbose:
        print("Model loaded")
    return model

def create_dataloader(args, cfg, ids, dataset):
    dataloader = DataLoader(
            ImageDataset(cfg, ids, dataset),
            batch_size=cfg.batch_size, num_workers=args.num_workers,
            drop_last=False, shuffle=False)
    return dataloader

def extractor(args):
    cfg = load_json(args.cfg_path)
    dataset = load_data(args, cfg)
    ids = get_ids(args, cfg, dataset)
    dataloader = create_dataloader(args, cfg, ids, dataset)
    model = get_model(args, cfg)

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    steps = len(dataloader)
    dataiter = iter(dataloader)
    feat_diffs = []
    for step in range(steps):
        print("Step: ", step)
        pos_imgs, neg_imgs = next(dataiter)
        if len(pos_imgs) <= 1:
            feat_diff = 0.0
        else:
            pos_feats  = [] 
            feat_diff = 0.0
            for image in pos_imgs:
                image = image.to(device)
                feat = model(image).squeeze(-1).squeeze(-1).detach().cpu().numpy()
                pos_feats.append(feat)
            # for image in neg_imgs:
            #     image = image.to(device)
            #     feat = model(image).squeeze(-1).squeeze(-1).detach().cpu().numpy()
            #     neg_feats.append(feat)
            
            for i in range(1,len(pos_feats)):
                feat_diff += np.linalg.norm((pos_feats[0], pos_feats[i]))
            feat_diff /= len(pos_imgs)
        feat_diffs.append(feat_diff)
    feat_mean = np.nanmean(feat_diffs)
    print(f"The feature difference mean for {cfg.Qcate} is: {feat_mean}")
    #save_file = os.path.join(args.save_path, f"{cfg.Qcate}.pt")
    #torch.save(feats, save_file)        
    
    





