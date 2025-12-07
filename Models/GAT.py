'ایاک النعبد و ایاک النستعین'

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

#GAT (Graph Attention Network)
class GATNet(nn.Module):
    def __init__(self, num_layer, heads, input_dim, hidden_dim, output_dim):
        super(GATNet, self).__init__()
        self.layers = nn.ModuleList()

        # اولین لایه: input_dim → hidden_dim
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads, concat=False))

        # لایه‌های میانی
        for _ in range(num_layer - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

        # لایه خروجی
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        x = self.lin(x)
        return x
