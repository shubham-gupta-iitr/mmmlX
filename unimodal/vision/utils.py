
import json, time, os, base64
import numpy as np
from collections import Counter
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from model import load_model, get_prediction
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
    query_type = cfg.Qcate
    ids = [id for id in dataset if dataset[id]["Qcate"] == query_type]
    if args.verbose:
        print(f"Total ids for query {query_type}: ", len(ids))
    return ids

def get_model(args, cfg):
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    model = load_model()
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
    feats = torch.Tensor()
    pred_boxes_list = []
    pred_class_list = []
    for step in range(steps):
        print("Step: ", step)
        image = next(dataiter)
        image = image.to(device)
        pred_boxes, pred_class = get_prediction(image, model)
        pred_boxes_list.append(pred_boxes)
        pred_class_list.append(pred_class)
    save_file = os.path.join(args.save_path, f"{cfg.Qcate}_boxes.pkl")
    with open(save_file, 'wb') as handle:
        pickle.dump(pred_boxes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    save_file = os.path.join(args.save_path, f"{cfg.Qcate}_class.pkl")
    with open(save_file, 'wb') as handle:
        pickle.dump(pred_class_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    





