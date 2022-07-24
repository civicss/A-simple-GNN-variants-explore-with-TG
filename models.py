import torch
import torch.nn.functional as F

from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.sage_conv import SAGEConv


# 定义模型
class GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, conv_type='GCN'):
        super().__init__()
        # 根据conv_type参数，选择对应的卷积层。由于框架的统一，不同卷积层有相同的初始化参数设置和接收的数据格式。
        if conv_type == 'GCN':
            CONV = GCNConv
        elif conv_type == 'GAT':
            CONV = GATConv
        elif conv_type == 'SAGE':
            CONV = SAGEConv
        else:
            raise NotImplemented('{} is not implied! Please chose from [GCN, GAT and SAGE]'.format(conv_type))

        self.conv1 = CONV(feature_dim, hidden_dim)
        self.conv2 = CONV(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        logits = self.conv2(h, edge_index)
        return logits
