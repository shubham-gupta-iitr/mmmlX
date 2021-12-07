
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv

 
# class ToyNet(nn.Module):
#     def __init__(self):
#         super(ToyNet, self).__init__()
#         num_features = 3584
#         num_classes = 1
#         self.conv1 = GCNConv(num_features, 2048)
#         self.conv2 = GCNConv(2048, 512)
#         self.conv3 = GCNConv(512, 256)
#         self.conv4 = GCNConv(256, 64)
#         self.conv5 = GCNConv(64, int(num_classes))
#         self._relu = nn.ReLU()
#         self._sigmoid = nn.Sigmoid()
#         self._dropout = nn.Dropout(0.3)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv2(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv3(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv4(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv5(x, edge_index)

        
#         return self._sigmoid(x)


embed_dim = 3584
import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import BatchNorm
class ToyNet(torch.nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()

        # self.conv1 = SAGEConv(embed_dim, 2048)
        self.conv1 = SAGEConv(512*3, 2048)
        self.bn1 = BatchNorm(2048)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(2048, 1024)
        self.bn2 = BatchNorm(1024)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(1024, 512)
        self.bn3 = BatchNorm(512)
        self.conv4 = SAGEConv(512, 256)
        self.bn4 = BatchNorm(256)
        self.conv5 = SAGEConv(256, 256)
        self.bn5 = BatchNorm(256)
        self.conv6 = SAGEConv(256, 128)
        self.bn6 = BatchNorm(128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +1, embedding_dim=embed_dim)
        # self.lin1 = torch.nn.Linear(256, 128)

        self.linq = torch.nn.Linear(768, 512)
        self.linc = torch.nn.Linear(768, 512)
        self.lini = torch.nn.Linear(2048, 512)

        self.linqci = torch.nn.Linear(512*3, 512)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        # self.lin3 = torch.nn.Linear(64, 1)
        self.lin3 = torch.nn.Linear(64, 2)

        self.blinq = torch.nn.BatchNorm1d(512)
        self.blinc = torch.nn.BatchNorm1d(512)
        self.blini = torch.nn.BatchNorm1d(512)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()  
        self._softmax = torch.nn.Softmax(dim=-1)      
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x.shape)
        x_ques = x[:,:768]
        x_cap = x[:,768:768*2]
        x_image = x[:,768*2:]

        x_ques = self.blinq(self.linq(x_ques))
        x_cap = self.blinc(self.linc(x_cap))
        x_image = self.blini(self.lini(x_image))

        x = torch.cat((x_ques,x_cap,x_image), dim=-1)
        # x = self.linqci(x)
        # print(x_ques.shape, x_cap.shape, x_image.shape)
        # x = self.item_embedding(x)
        # x = x.squeeze(1)        
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        # print(x.shape)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # print(x1.shape)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
     
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.relu(self.bn5(self.conv5(x, edge_index)))
        x = F.relu(self.bn6(self.conv6(x, edge_index)))

        # x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        # x = torch.sigmoid(self.lin3(x)).squeeze(1)
        x_log = self.lin3(x)
        x = self._softmax(x_log)
        # print(x.shape)
        # x = torch.sigmoid(self.lin3(x)).squeeze(1)
        return x_log, x 