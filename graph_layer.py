import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 inter_dim=-1, **kwargs):  # inter_dim = 128
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels  # train_config['slide_win']
        self.out_channels = out_channels    # 64
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        # self.dp = nn.Dropout(dropout)

        self.__alpha__ = None

        self.lin = Linear(self.in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))    # 四个attention 的线性层
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):  # 每次调用GraphLayer都重新初始化参数
        glorot(self.lin.weight)  # Xavier初始化(glorot初始化)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        # print("graph_layer_x.shape_forward_before: ", x.shape, embedding.shape)  # torch.Size([224, 48])   32x7=224 batch_size*node_num  torch.Size([224, 64])
        if torch.is_tensor(x):
            x = self.lin(x)
            # print("x.shape", x.shape)  # torch.Size([224, 64])
            x = (x, x)
            # print("aaaaaa", x)  # 将两个x tensor组合在一起
        else:
            x = (self.lin(x[0]), self.lin(x[1]))
        # 先移除所有的自循环节点，再为每一个节点增加一个自循环，表明每个节点都与自身相关，并且没有重复的自循环
        # print("graph_layer_edge_index", edge_index)
        edge_index, _ = remove_self_loops(edge_index)   # 移除edge_index中孤立的节点，即与其他节点不相连的点
        # print("graph_layer_remove_self_loops", edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))  # Adds a self-loop (i,i)∈E to every node i∈V in the graph given by edge_index.
        # print("graph_layer_add_self_loops", edge_index)  # 每次反向传播都会改变

        # propagate接收边缘索引和其他可选信息，例如节点特征（嵌入）。因此，调用此函数将调用消息并进行更新。
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)
        # print("graph_layer", out)
        # print("graph_layer_out.shape: ", out.shape)  # torch.Size([224, 1, 64])

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            # print("concat_out.shape: ", out.shape)
        else:
            out = out.mean(dim=1)
            # print("mean_out.shape: ", out.shape)    # torch.Size([224, 64])

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            # print("graph_layer_alpha: ", out.shape, alpha.shape)    # torch.Size([224, 64])   torch.Size([672, 1, 1])
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights):
        '''
        您可以指定为每个节点对（x_i，x_j）构造“消息”的方式。由于它遵循传播的调用，它可以传递任何参数传播。
        需要注意的一点是，您可以使用“_i”和“_j”定义从参数到特定节点的映射。因此，在命名此函数的参数时必须非常小心。
        '''
        # print("graph_layer_x_j.shape_before: ", x_j.shape)  # torch.Size([672, 64])
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        # print("graph_layer_x_j.shape_x_i_x_j_after: ", x_j.shape, embedding.shape, x_i, x_j)   # torch.Size([672, 1, 64])  torch.Size([224, 64]) embedding [7,64]重复b_s次变为[32*7,64]=[224,64]

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            # print("graph_layer:edge_index_i, edges[0]", edge_index_i, edges[1], edges[0]) # edge_index_i==edge[1] target
            # 每次反向传播都会改变 edge_index_i为所有特征部分的index edge_index[1](target)，edges[0]为topk部分的index  edge_index[0](source)
            # print("embedding[edge_index_i]", embedding_i.shape, embedding[edge_index_i].shape, embedding[edge_index_i])     # torch.Size([672, 64])torch.Size([672, 64])
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)
            # print("graph_layer_edge_index_i, edges, edges[0], embedding_i, embedding_j", edge_index_i, edges, edges[0], embedding_i, embedding_j)
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)
            # print("embedding_i, x_i", embedding_i, x_i)
            # print("key_i.shape", key_i.shape)   # key_i.shape torch.Size([672, 1, 128])

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        # print("alpha.shape_1", alpha.shape, alpha)  # torch.Size([672, 1])

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)  # Graph Attention-Based Forecasting - Feature Extractor
        # print("alpha.shape", size_i, alpha.shape, alpha)   # size_i = 224, torch.Size([672, 1, 1]) index 相同的部分做 softmax，然后获得每条边的权值，与 x_j 相乘得到最终 embedding

        if return_attention_weights:
            self.__alpha__ = alpha
        # print("self.__alpha__, self.__alpha__.shape", self.__alpha__, self.__alpha__.shape)  # [672, 1, 1]
        # print("x_i * alpha.view(-1, self.heads, 1)", x_i * alpha.view(-1, self.heads, 1), x_j * alpha.view(-1, self.heads, 1))
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # alpha = self.dp(alpha)

        # print("graph_layer_x_j.shape, ", x_j, x_j.shape)    # torch.Size([672, 1, 64])
        # print("graph_layer_x_j * alpha.view(-1, self.heads, 1): ", x_j * alpha.view(-1, self.heads, 1), (x_j * alpha.view(-1, self.heads, 1)).shape) # torch.Size([672, 1, 64])
        return x_j * alpha.view(-1, self.heads, 1)  # 计算出注意力系数alpha  torch.Size([672, 1, 64])

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
