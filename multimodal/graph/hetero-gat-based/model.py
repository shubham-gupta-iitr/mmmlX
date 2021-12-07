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
#from torch_geometric.nn import TransformerConv as GATConv
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels=3584, out_channels=2):
#         super().__init__()
#         in_channels = 3584
#         out_channels = 2
#         self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
#         # On the Pubmed dataset, use heads=8 in conv2.
#         self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
#                              dropout=0.6)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
#         #return F.log_softmax(x, dim=-1)

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 512
        self.in_head = 8
        self.out_head = 1
        num_classes = 2
        self.conv1 = GATConv((-1,-1), self.hid, heads=self.in_head,
                            concat=True)
        self.conv5 = GATConv((-1,-1), num_classes, concat=False,
                             heads=self.out_head)
        # TODO (Nikhil, Saloni) -> Apply Edge loss on a different Attention head..

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv5(x, edge_index)
        return x
        