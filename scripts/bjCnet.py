import torch.nn.functional as F
from torch import nn
import torch
from dgl.nn.pytorch import GatedGraphConv, GraphConv

device = torch.device('cuda:0')

class MyHingeLoss(nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, vector_A, vector_B, vector_C):
        # vector_A->bug,1
        # vector_B->patch,0
        # vector_C->clean,0
        loss_func = torch.nn.CrossEntropyLoss()
        labels = torch.ones(vector_A.shape[0], dtype=torch.long).to(device)
        loss_A = loss_func(vector_A, labels)

        labels = torch.zeros(vector_B.shape[0], dtype=torch.long).to(device)
        loss_B = loss_func(vector_B, labels)

        labels = torch.zeros(vector_C.shape[0], dtype=torch.long).to(device)
        loss_C = loss_func(vector_C, labels)

        loss = (loss_A + loss_B + loss_C) / 3
        return loss


class MakeModule:
    @staticmethod
    def make_GatedGCN(in_feats, out_feats, n_etypes, n_steps=5):
        return GatedGraphConv(in_feats, out_feats, n_steps, n_etypes)

    @staticmethod
    def make_GraphConv(in_feats, out_feats):
        return GraphConv(in_feats, out_feats, norm='both', weight=True, bias=True, allow_zero_in_degree=True)


class GraphConvNet(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_layers):
        super(GraphConvNet, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(MakeModule.make_GraphConv(in_feats, hidden_dim))
            else:
                self.layers.append(MakeModule.make_GraphConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, 2)
        )

    def forward(self, g):
        gx = []
        for conv, bn in zip(self.layers, self.batch_norms):
            x = conv(g, g.ndata["X"])
            x = F.relu(x)
            x = bn(x)
            gx.append(x)

        graph_features = gx[0]
        for idx in range(1, len(gx)):
            graph_features = torch.cat((graph_features, gx[idx]), dim=1)

        return graph_features

class GatedGCN(nn.Module):
    def __init__(self, in_feats, n_etypes, hidden_dim, num_layers):
        super(GatedGCN, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(MakeModule.make_GatedGCN(in_feats, hidden_dim, n_etypes))
            else:
                self.layers.append(MakeModule.make_GatedGCN(hidden_dim, hidden_dim ,n_etypes))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, 2)
        )

    def forward(self, g):
        gx = []
        for conv, bn in zip(self.layers, self.batch_norms):
            x = conv(g, g.ndata["X"], g.edata["X"])
            x = F.relu(x)
            x = bn(x)
            gx.append(x)

        graph_features = gx[0]
        for idx in range(1, len(gx)):
            graph_features = torch.cat((graph_features, gx[idx]), dim=1)

        return graph_features


class Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, g):
        g = g.to(device)
        g_feats = self.encoder(g)
        g.ndata["X"] = g_feats

        return g