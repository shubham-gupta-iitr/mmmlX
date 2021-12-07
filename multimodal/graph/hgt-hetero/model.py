import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
#from torch_geometric.nn.conv.gatv2_conv import GATv2Conv as GATConv

# from torch_geometric.nn import SuperGATConv as GATConv
# from torch_geometric.nn import 
from torch_geometric.nn.norm import LayerNorm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
#from torch_geometric.nn.conv.gatv2_conv import GATv2Conv as GATConv
from torch_geometric.nn import TransformerConv as GATConv
from torch_geometric.nn import HGTConv

class GAT(nn.Module):
    def __init__(self, meta):
        super(GAT, self).__init__()
        self.hid = 512
        self.in_head = 8
        self.out_head = 1
        num_classes = 2
        self.conv1 = HGTConv(-1, self.hid, meta, heads=self.in_head)
        self.conv5 = HGTConv(-1, num_classes, meta,
                             heads=self.out_head)
        # TODO (Nikhil, Saloni) -> Apply Edge loss on a different Attention head..

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv5(x, edge_index)
        return x
        