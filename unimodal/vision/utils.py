
import json, time, os, base64
import numpy as np
from collections import Counter
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from model import Model
from dataset import ImageDataset
from easydict import EasyDict as edict
import pickle

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
    ids = [id for id in dataset if dataset[id]["Qcate"] == cfg.Qcate]
    query_type = cfg.Qcate
    if args.verbose:
        print(f"Total ids for query {query_type}: ", len(ids))
    return ids

def get_model(args, cfg):
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    model = Model(cfg, device)
    #model = DataParallel(model, device_ids=device_ids).to(device).eval()
    if args.verbose:
        print("Model loaded")
    return model

def create_dataloader(args, cfg, ids, dataset, processor):
    dataloader = DataLoader(
            ImageDataset(cfg, ids, dataset, processor),
            batch_size=cfg.batch_size, num_workers=args.num_workers,
            drop_last=False, shuffle=False)
    return dataloader

def extractor(args):
    cfg = load_json(args.cfg_path)
    dataset = load_data(args, cfg)
    ids = get_ids(args, cfg, dataset)
    model = get_model(args, cfg)
    dataloader = create_dataloader(args, cfg, ids, dataset, model.preprocess)

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    steps = len(dataloader)
    dataiter = iter(dataloader)
    feats = torch.Tensor()
    image_ids = []
    for step in range(steps):
        print("Step: ", step)
        image, image_id = next(dataiter)
        image = image.to(device)
        feat = model(image).squeeze(-1).squeeze(-1).detach().cpu()
        feats = torch.cat((feats, feat), 0)
        image_ids.extend(image_id)
    save_file = os.path.join(args.save_path, f"{cfg.Qcate}.pt")
    torch.save(feats, save_file)
    save_file = os.path.join(args.save_path, f"{cfg.Qcate}_image_ids.pkl")
    with open(save_file, 'wb') as handle:
        pickle.dump(image_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
    





