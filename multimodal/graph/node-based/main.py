from numpy.core.numeric import outer
from scipy.sparse import data
import torch
from dataset import WebQnaDataset
from torch_geometric.loader import DataLoader
import numpy as np
from model import ToyNet
from sklearn.metrics import f1_score
data_root = "/home/ubuntu/WebQna/nodes-2611"
batch_size = 32
epochs = 100
device = "cuda"

print_step = 270
val_step = 500

webqa_dataset_train = WebQnaDataset(data_root)
webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_train = DataLoader(webqa_dataset_train, batch_size, shuffle=True)
webqa_dataloader_val = DataLoader(webqa_dataset_val, batch_size, shuffle=True)
# key = torch.tensor([10])
# d = webqa_dataset.get(idx=10)

toy_model = ToyNet().to(device)

# criterion = torch.nn.BCELoss()
class_weights = torch.tensor([1,4], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.AdamW(toy_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
toy_model.train()
 
g_ctr = 0
for epoch in range(epochs):
    g_ctr = 0
    for idx, datum in enumerate(webqa_dataloader_train):
        g_ctr += 1
        toy_model.train()
        optimizer.zero_grad()
        
        datum = datum.to(device)
        outp, pred = toy_model(datum)
        # print(outp.shape)
        # print(datum)
        # outp = outp.squeeze(1)
        # print(datum.y.shape, outp.shape)
        loss = criterion(outp, datum.y)
        loss.backward()
        optimizer.step()
        if g_ctr % print_step == 0:
            print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), scheduler.get_last_lr()[0]))
        
        if g_ctr % val_step == 0:
            toy_model.eval()
            val_pred = 0.
            total_vals = 0.
            all_gt = []
            all_pred = []
            for idx, datum_val in enumerate(webqa_dataloader_val):
                total_vals += datum_val.x.shape[0]
                datum_val = datum_val.to(device)
                _,pred = toy_model(datum_val)
                # outp = outp.squeeze(1)
                # outp = outp >0.5
                # print(pred.shape)
                pred = torch.argmax(pred, dim=-1)
                # gt = torch.argmax(datum_val.y, dim=-1)
                gt = datum_val.y
                # print(datum_val.x.shape[0],torch.sum(outp==0))
                # val_pred += torch.sum(outp==datum_val.y).detach().cpu().item()
                val_pred += torch.sum(pred==gt).detach().cpu().item()
                # all_gt.extend(datum_val.y.detach().cpu())
                all_gt.extend(gt.detach().cpu())
                all_pred.extend(pred.detach().cpu())
            f1s = f1_score(all_gt,all_pred)
            print("Epoch :: {} Step :: {} Val Acc :: {}  F1 Score :: ".format(epoch, g_ctr, (val_pred/total_vals)*100), f1s)
    scheduler.step()