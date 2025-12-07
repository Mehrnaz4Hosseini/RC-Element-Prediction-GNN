'ایاک النعبد و ایاک النستعین'

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, num_layer, heads, input_dim, hidden_dim, output_dim):
        super(GraphTransformer, self).__init__()

        self.layers = nn.ModuleList()

         # خروجی هر لایه با concat
        out_dim = hidden_dim * heads 

        # لایه اول: input_dim → hidden_dim * heads
        self.layers.append(TransformerConv(input_dim, hidden_dim, heads=heads, concat=True))

        # لایه‌های میانی: hidden_dim * heads → hidden_dim * heads
        for _ in range(num_layer - 1):
            self.layers.append(TransformerConv(out_dim , hidden_dim, heads=heads, concat=True))

        # لایه نهایی خطی: hidden_dim * heads → output_dim
        self.lin = nn.Linear(out_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        x = self.lin(x)
        return x

#class GraphTransformer(nn.Module):
    #def __init__(self,heads, input_dim, hidden_dim, output_dim):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads, concat=False)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads, concat=False)
        self.lin = nn.Linear(hidden_dim, output_dim)

    #def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return self.lin(x)
