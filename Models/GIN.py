'ایاک النعبد و ایاک النستعین'
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

#GIN (Graph Isomorphism Network)
class GINNet(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim, output_dim):
        super(GINNet, self).__init__()
        self.layers = nn.ModuleList()

        # تعریف تابع MLP برای هر لایه GIN
        def build_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        # لایه اول
        self.layers.append(GINConv(build_mlp(input_dim, hidden_dim)))

        # لایه‌های میانی
        for _ in range(num_layer - 1):
            self.layers.append(GINConv(build_mlp(hidden_dim, hidden_dim)))

        # لایه خروجی
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        x = self.lin(x)
        return x
