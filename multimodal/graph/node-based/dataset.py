import os.path as osp
import typing
from numpy.lib.utils import source

import torch
# from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from utils import get_ids, load_bert_feats, load_data, load_image_feats
import torch_geometric.transforms as T

class WebQnaDataset(Dataset):
    def __init__(self, root, val=False, transform=None, pre_transform=None):
        self._WebQna_dataset = load_data()
        self._question_ids = get_ids(self._WebQna_dataset, val=val)
        self._processed_file_names = ["node_"+str(q)+".pt" for q in self._question_ids]
        
        self._bert_feats = load_bert_feats()
        self._image_feats = load_image_feats()
                
        self._caption_feats = {}
        self._question_feats = {}
        self._question_ids = self._question_ids[:18000]
        t_id = self._question_ids[-1]
        t2_id = self._question_ids[0]
        for id in self._question_ids:
            for k in self._bert_feats[id].keys():
                if k=='Q':
                    self._question_feats[id] = torch.tensor(self._bert_feats[id]['Q'])
                elif 'img' in k:
                    for img_k in self._bert_feats[id][k].keys():
                        self._caption_feats[img_k] = torch.tensor(self._bert_feats[id][k][img_k])

        # print(self._question_feats[t_id])      
        # print(self._question_feats[t2_id])      
        # print(self._WebQna_dataset[t_id]['Q'])
        # print(t_id)
        # for id in self._question_feats:
        #     if len(self._WebQna_dataset[id]['img_posFacts']) > 1:
        #         print(self._WebQna_dataset[id]['img_posFacts'])
        #         break
        
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self._processed_file_names
        # return []
    def len(self):
        return len(self._question_ids)
        pass
    
    def get(self, idx:int)-> Data:
        y = []
        x = []
        id = self._question_ids[idx]
        ques_feat = self._question_feats[id]
        for pos_image_dict in self._WebQna_dataset[id]['img_posFacts']:
            image_id = pos_image_dict['image_id']
            pos_image_feat = self._image_feats[image_id]
            pos_image_caption_feat = self._caption_feats[image_id]
            # node_features = torch.cat((ques_feat, pos_image_feat, pos_image_caption_feat),dim=-1).unsqueeze(0)
            node_features = torch.cat((ques_feat, pos_image_caption_feat, pos_image_feat),dim=-1).unsqueeze(0)
            x.append(node_features)
            y.append(1)
            # y.append([0,1])
        for neg_image_dict in self._WebQna_dataset[id]['img_negFacts']:
            image_id = neg_image_dict['image_id']
            neg_image_feat = self._image_feats[image_id]
            neg_image_caption_feat = self._caption_feats[image_id]
            # node_features = torch.cat((ques_feat, neg_image_feat, neg_image_caption_feat),dim=-1).unsqueeze(0)
            node_features = torch.cat((ques_feat, neg_image_caption_feat, neg_image_feat),dim=-1).unsqueeze(0)
            x.append(node_features)
            y.append(0)
            # y.append([1,0])
        # for sh in x:
        #     # print(sh[0].shape[0])
        #     if sh[0].shape[0] != 768:
        #         print("missing q")
        #         break
        #     if sh[2].shape[0] != 768:
        #         print("missing cap")
        #         break
        
        node_idx = [i for i in range(len(y))]
        # node_idx_idx = np.arange(start=0,stop=len(y),step=1)
        # node_idx = list(np.random.permutation(node_idx))
        
        # print(node_idx)
        source_nodes = []
        for i in range(len(y)):
            source_nodes += [i]*(len(y)-1)
        target_nodes = []
        for i in range(len(y)):
            target_nodes += node_idx[:i] + node_idx[i+1:]
        
        # source_nodes = node_idx[:-1]
        # target_nodes = node_idx[1:]
        # print(len(source_nodes), len(target_nodes))
        # assert False
        # edge_index = torch.tensor([source_nodes + target_nodes, target_nodes + source_nodes], dtype=torch.long)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        x = torch.cat(x,dim=0)
        # y = torch.FloatTensor(y)
        y = torch.LongTensor(y)
        # y = torch.IntTensor(y)
        data = Data(x=x, edge_index=edge_index, y=y)
        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        data = T.NormalizeFeatures()(data)
        return data

    # def process(self):

    #     for id in tqdm(self._question_ids):
    #         # y = []
    #         # x = []
    #         # ques_feat = torch.tensor(self._question_feats[id])
    #         # for pos_image_dict in self._WebQna_dataset[id]['img_posFacts']:
    #         #     image_id = pos_image_dict['image_id']
    #         #     pos_image_feat = self._image_feats[image_id]
    #         #     pos_image_caption_feat = self._caption_feats[image_id]
    #         #     node_features = [ques_feat,pos_image_feat, pos_image_caption_feat]
    #         #     x.append(node_features)
    #         #     y.append(1)
    #         # for pos_image_dict in self._WebQna_dataset[id]['img_negFacts']:
    #         #     image_id = pos_image_dict['image_id']
    #         #     pos_image_feat = self._image_feats[image_id]
    #         #     pos_image_caption_feat = self._caption_feats[image_id]
    #         #     node_features = [ques_feat,pos_image_feat, pos_image_caption_feat]
    #         #     x.append(node_features)
    #         #     y.append(0)
    #         # for sh in x:
    #         #     # print(sh[0].shape[0])
    #         #     if sh[0].shape[0] != 768:
    #         #         print("missing q")
    #         #         break
    #         #     if sh[2].shape[0] != 768:
    #         #         print("missing cap")
    #         #         break
    #         # node_idx = [i for i in range(len(y))]
    #         # source_nodes = node_idx[:-1]
    #         # target_nodes = node_idx[1:]
    #         # edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            
    #         # y = torch.FloatTensor(y)
    #         # data = Data(x=x, edge_index=edge_index, y=y)
            
    #         # torch.save(data, osp.join(self.processed_dir, f'node_{id}.pt'))
