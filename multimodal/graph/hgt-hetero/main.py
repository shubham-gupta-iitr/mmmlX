from numpy.core.numeric import outer
from scipy.sparse import data
import torch
from dataset import WebQnaDataset
from torch_geometric.loader import DataLoader
import numpy as np
from model import GAT
from sklearn.metrics import f1_score
#from torch_geometric.nn import to_hetero
import torch
from torch.nn import ReLU
import torch.nn.functional as F

from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero
from torch.optim.lr_scheduler import _LRScheduler

class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


data_root = "/home/ubuntu/WebQna/nodes-2611"
batch_size = 64
epochs = 500
device = "cuda"

print_step = 270
val_step = 500

webqa_dataset_train = WebQnaDataset(data_root)
webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_train = DataLoader(webqa_dataset_train, batch_size, shuffle=True)
webqa_dataloader_val = DataLoader(webqa_dataset_val, batch_size, shuffle=True)
# key = torch.tensor([10])
# d = webqa_dataset.get(idx=10)



# model = Sequential('x, edge_index', [
#     (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (Linear(-1, 2), 'x -> x'),
# ])
# toy_model = model
graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','link3','ques'), ('img_src','link4','ques')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src'), 
#('txt_src','link3','img_src'), ('img_src','link4','txt_src')])
toy_model = GAT(graph_meta)
#toy_model = to_hetero(toy_model, graph_meta)
toy_model = toy_model.to(device)

# criterion = torch.nn.BCELoss()
class_weights = torch.tensor([1,10], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.AdamW(toy_model.parameters(), lr=0.0001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
scheduler = NoamLR(optimizer, warmup_steps=10)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, threshold=0.01, min_lr=1e-6)
toy_model.train()
 
g_ctr = 0
for epoch in range(epochs):
    g_ctr = 0
    for idx, datum in enumerate(webqa_dataloader_train):
        g_ctr += 1
        toy_model.train()
        optimizer.zero_grad()
        
        datum = datum.to(device)
        #print(datum)
        #assert(False)
        #outp, pred = toy_model(datum.x_dict, datum.edge_index_dict)
        outp = toy_model(datum.x_dict, datum.edge_index_dict)
        #print(outp)
        #print(datum)
        # outp = outp.squeeze(1)
        # print(datum.y.shape, outp.shape)
        #loss = criterion(outp, datum.y_dict)
        loss_img = criterion(outp['img_src'], datum.y_dict['img_src'])
        loss_txt = criterion(outp['txt_src'], datum.y_dict['txt_src'])
        loss = loss_img + loss_txt
        loss.backward()
        optimizer.step()
        if g_ctr % print_step == 0:
            #print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), scheduler.get_last_lr()[0]))
            print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), optimizer.param_groups[0]['lr']))
        
        if g_ctr % val_step == 0:
            toy_model.eval()
            val_pred = 0.
            total_vals = 0.
            all_gt = []
            all_pred = []
            for idx, datum_val in enumerate(webqa_dataloader_val):
                #total_vals += datum_val.x.shape[0]
                total_vals += datum_val.x_dict['img_src'].shape[0]
                total_vals += datum_val.x_dict['txt_src'].shape[0]
                datum_val = datum_val.to(device)
                pred = toy_model(datum_val.x_dict, datum_val.edge_index_dict)
                loss_img_val = criterion(pred['img_src'], datum_val.y_dict['img_src'])
                loss_txt_val = criterion(pred['txt_src'], datum_val.y_dict['txt_src'])
                loss_val = loss_img_val + loss_txt_val
                # outp = outp.squeeze(1)
                # outp = outp >0.5
                # print(pred.shape)
                #pred = torch.argmax(pred, dim=-1)
                pred_img = torch.argmax(pred['img_src'], dim=-1)
                pred_txt = torch.argmax(pred['txt_src'], dim=-1)
                # gt = torch.argmax(datum_val.y, dim=-1)
                #gt = datum_val.y
                gt_img = datum_val.y_dict['img_src']
                gt_txt = datum_val.y_dict['txt_src']
                # print(datum_val.x.shape[0],torch.sum(outp==0))
                # val_pred += torch.sum(outp==datum_val.y).detach().cpu().item()
                #val_pred += torch.sum(pred==gt).detach().cpu().item()
                val_pred_img = torch.sum(pred_img==gt_img).detach().cpu().item()
                val_pred_txt = torch.sum(pred_txt==gt_txt).detach().cpu().item()
                val_pred += val_pred_img + val_pred_txt
                # all_gt.extend(datum_val.y.detach().cpu())
                #all_gt.extend(gt.detach().cpu())
                #all_pred.extend(pred.detach().cpu())
                
                all_gt.extend(gt_img.detach().cpu())
                all_gt.extend(gt_txt.detach().cpu())
                all_pred.extend(pred_img.detach().cpu())
                all_pred.extend(pred_txt.detach().cpu())
            f1s = f1_score(all_gt,all_pred)
            print("Epoch :: {} Step :: {} Val Acc :: {}  F1 Score :: ".format(epoch, g_ctr, (val_pred/total_vals)*100), f1s)
    #scheduler.step(loss_val)
    scheduler.step()