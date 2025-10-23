import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_adj
from src.models.components.prediction_model import MLPNet
from src.utils import get_activation
from typing import Optional, List


class GAT(torch.nn.Module):
    def __init__(self,
                 x_dim: int,
                 hidden_channels: Optional[List[int]],
                 out_channel: int,
                 edge_dim: int,
                 activation: str = "relu",
                 heads: int = 1,
                 conv_dropout: float = 0.0,
                 node_dropout: float = 0.0,
                 edge_dropout: float = 0.0,
                 add_self_loops: bool = True,
                 fill_value: str = "max",
                 share_weights: bool = False,
                 output_activation: str = "none",
                 normalize: bool = False,
                 post_mlp_hiddens: Optional[List[int]] = None,
                 post_activation: str = "relu",
                 post_input_dropout: float = 0.0,
                 post_dropout: float = 0.0,
                 use_post_mlp: bool = False):
        super().__init__()

        assert(post_input_dropout <= 1e-4 or node_dropout <= 1e-4)  # both are not positive simultaneously

        if post_mlp_hiddens is None:
            post_mlp_hiddens = []

        self.convs = torch.nn.ModuleList()
        self.conv_posts = torch.nn.ModuleList()

        if hidden_channels is None:
            hidden_channels = []

        for i, hidden_channel in enumerate(hidden_channels):
            # perform convolution
            self.convs.append(GATv2Conv(x_dim, hidden_channel, add_self_loops=add_self_loops, fill_value=fill_value,
                                        edge_dim=edge_dim, heads=heads, dropout=conv_dropout,
                                        share_weights=share_weights, concat=True))
            x_dim = heads * hidden_channel


            layers = []
            # norm
            if normalize:
                layers.append(torch.nn.LayerNorm(heads * hidden_channel))
            # activate
            layers.append(get_activation(activation))
            # dropout
            layers.append(torch.nn.Dropout(p=node_dropout))

            self.conv_posts.append(torch.nn.Sequential(*layers))

        if use_post_mlp:
            self.post_mlp = MLPNet(heads * hidden_channels[-1], out_channel, hidden_layer_sizes=post_mlp_hiddens,
                                   hidden_activation=post_activation, dropout=post_dropout,
                                   output_activation=output_activation, input_dropout=post_input_dropout)
        else:
            self.post_mlp = None
            self.convs.append(GATv2Conv(heads * hidden_channels[-1], out_channel, add_self_loops=add_self_loops,
                                        fill_value=fill_value, edge_dim=edge_dim, heads=heads, dropout=conv_dropout,
                                        share_weights=share_weights, concat=False))

        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_attr, stage="train"):
        training = True if stage == "train" else False
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.edge_dropout, training=training)

        for conv, conv_post in zip(self.convs, self.conv_posts):
            x = conv(x, edge_index, edge_attr)

            if conv_post is not None:
                x = conv_post(x)

        if self.post_mlp is not None:
            x = self.post_mlp(x)
        else:
            x = self.convs[-1](x, edge_index, edge_attr)

        return x
