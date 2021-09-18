import numpy as np
import torch
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]  # 21 3x7-topK*node_num
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    # print("GDN_edge_index, edge_num", edge_index, edge_num)  # edge_index与batch_edge_index相同，都是GDN中生成的gated_edge_index
    # print("GDN_batch_edge_index_all", batch_edge_index, batch_edge_index.shape)  # torch.Size([2, 672]  21*batch_size32 = 672
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
        # print("GDN_batch_edge_index:", i, batch_edge_index, i * node_num)
    # print("GDN_batch_edge_index, batch_edge_index.shape", batch_num, batch_edge_index, batch_edge_index.shape)  # torch.Size([2, 960]) 32*30=960
    return batch_edge_index.long()


class OutLayer(nn.Module):  # 多层Linear+BN+ReLU层堆叠得到的mlp,最后一层为Linear层
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)  # 48, 64, 128

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        # print("Attention_TCN_out.shape", out.shape)  # torch.Size([224, 64])
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        # print("self.att_weight_1:", self.att_weight_1)

        out = self.bn(out)

        return self.relu(out)  # 即为原文中的z_i^(t)


class AR(nn.Module):

    def __init__(self, window, n_multiv):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)  # （120,1）
        self.out_linear = nn.Linear(n_multiv, 1)

    def forward(self, x):
        # print(x.shape)  # torch.Size([16, 7, 96])
        x = self.linear(x)
        # print("aaaaaaaaaaaaa", x.shape)  # torch.Size([16, 7, 1])
        x = torch.transpose(x, 1, 2)
        # print("aaaaaaaaaaaaa", x.shape)  # torch.Size([16, 1, 7])
        x = self.out_linear(x)  ##
        # print("aaaaaaaaaaaaa", x.shape)  # torch.Size([16, 1, 1])
        x = torch.squeeze(x, 2)  ##
        # print("aaaaaaaaaaaaa", x.shape)  # torch.Size([16, 1])
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        # self.kernel_set = [2, 3, 6, 7]
        self.kernel_set = [2, 6]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
        # print("dilation_factor: ", dilation_factor)

    def forward(self, input):
        # print("dilated_inception, input.shape", input.shape, input)     # torch.Size([64, 32, 6, 96])
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
            # print("x, len(x)", self.tconv[i](input).shape, i, len(x))   # 0，1    1，2    2，3    3，4
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
            # print("x, len(x)", i, len(x), -x[-1].size(3))   # 0，4  -58 -52 -46    1，4   -58 -52 -46    2，4  -58 -52 -46     3，4   -58 -52 -46
        x = torch.cat(x, dim=1)
        # print("x, x.shape", x.shape)  # ([32, 32, 6, 58])    ([32, 32, 6, 52])   ([32, 32, 6, 46])
        return x


class Temporal(nn.Module):
    def __init__(self, conv_channels=32, residual_channels=32, layers=3, in_dim=1, out_dim=64, dilation_exponential=1):
        super(Temporal, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.dp = nn.Dropout(0.3)
        new_dilation = 1
        for j in range(1, layers + 1):
            self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
            self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
            new_dilation *= dilation_exponential
        self.layers = layers
        self.end_conv = nn.Conv2d(in_channels=conv_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.mlp = nn.Linear(49, 64)  # 如果为[2,6]，则为（33，64），如果为[2,7]，则为（46，64）

    def forward(self, input):
        # print("GDN_TCN_input.shape:", input.shape)  # torch.Size([32, 1, 7, 64])
        x = self.start_conv(input)  # torch.Size([32, 32, 7, 64])
        # print(x.shape)
        # x = input
        for i in range(self.layers):
            filter = self.filter_convs[i](x)
            # print("GDN_TCN_filter.shape:", filter.shape)    # torch.Size([32, 32, 7, 58])  torch.Size([32, 32, 7, 52])  torch.Size([32, 32, 6, 46])
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            # print("GDN_TCN_gate.shape:", gate.shape)    # torch.Size([32, 32, 7, 58])  torch.Size([32, 32, 7, 52])  torch.Size([32, 32, 7, 46])
            gate = torch.sigmoid(gate)
            x = filter * gate
            # print("GDN_TCN_x.shape:", x.shape)  # torch.Size([32, 32, 7, 58])  torch.Size([32, 32, 7, 52])  torch.Size([32, 32, 7, 46])
            x = self.dp(x)
            # print("GDN_TCN_dpx.shape:", x.shape)  # torch.Size([32, 32, 7, 58])  torch.Size([32, 32, 7, 52])  torch.Size([32, 32, 7, 46])

        x = self.end_conv(x)
        # print("GDN_TCN_end_convx.shape:", x.shape)  # torch.Size([32, 1, 7, 46])
        x = self.mlp(x)
        # print("GDN_TCN_mlp.shape:", x.shape)  # torch.Size([32, 1, 7, 64])
        return x


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1,
                 topk=20, conv_channels=32, residual_channels=32, layers=3, in_dim=1, out_dim=1,
                 dilation_exponential=1):
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)

        edge_set_num = len(edge_index_sets)
        # print("GDN_edge_set_num_edge_index_sets: ", edge_set_num, edge_index_sets)   # 1
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)  # 10, 64, 128
        ])

        # self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.temp_layer = Temporal(conv_channels=conv_channels, residual_channels=residual_channels, layers=layers,
                                   in_dim=in_dim, out_dim=out_dim, dilation_exponential=dilation_exponential)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.ar = AR(window=input_dim, n_multiv=node_num)

        # self.dp = nn.Dropout(0.2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.mlp = nn.Linear(node_num, 1)
        # print("GDN_batch_num, node_num", batch_num, node_num) # 32 6

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        x_raw = data.clone().detach()
        # print("GDN_AR_Attention_TCN:org_edge_index", org_edge_index) #原始的org_edge_index并没有用，采用了模型自己学习出来的graph
        # tcn_input = data.clone().detach()   # torch.Size([32, 7, 48])
        # print("GDN_tcn_input.shape: ", tcn_input.shape)
        # tcn_input = tcn_input.reshape(tcn_input.shape[0], 1, tcn_input.shape[1], tcn_input.shape[2])    # torch.Size([32, 1, 6, 96])
        # print("GDN_tcn_input.shape: ", tcn_input.shape)
        edge_index_sets = self.edge_index_sets  # 原始的edge_index所有候选集

        device = data.device

        batch_num, node_num, all_feature = x_raw.shape
        # print("GDN_x.shape", x_raw.shape, x_raw)   # torch.Size([32, 7, 48])
        x = x_raw.view(-1, all_feature).contiguous()  # torch.Size([224, 48])

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            # print("GDN_i, edge_index, edge_index_sets", i, edge_index, edge_index_sets) # i = 0 两个tensor相同，都是main中生成的fc_edge_index

            all_embeddings = self.embedding(torch.arange(node_num).to(device))  # Sensor Embedding
            # print("GDN_all_embeddings, all_embeddings.shape", all_embeddings, all_embeddings.shape) # torch.Size([6, 64])
            weights_arr = all_embeddings.detach().clone()  # Sensor Embedding
            all_embeddings = all_embeddings.repeat(batch_num, 1)  # Sensor Embedding

            # print("all_embeddings.shape", all_embeddings.shape)  # torch.Size([224, 64])

            weights = weights_arr.view(node_num, -1)
            # print("GDN_weights.shape: ", weights, weights.shape)  # torch.Size([7, 64])
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat  # Compute e_ji  cos_ji_mat.shape:[7, 7]

            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]   #　dim=1是按行取值 torch.topk[0]为值，torch.topk[1]为indice
            # print("GDN_cos_ji_mat_topk_indices_ji_topk_indices_ji.shape: ", cos_ji_mat, topk_indices_ji, topk_indices_ji.shape)  # torch.Size([7, 3])

            self.learned_graph = topk_indices_ji  # Graph Structure Learning

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            # print("GDN_gated_i, gated_j, gated_edge_index, topk_indices_ji:", gated_i, gated_j, gated_edge_index)
            '''tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
         4, 5, 5, 5, 5, 5]], device='cuda:0') tensor([[0, 3, 4, 5, 2, 1, 2, 4, 3, 5, 2, 3, 4, 5, 1, 3, 2, 4, 0, 1, 4, 3, 2, 1,
         0, 5, 2, 1, 0, 4]], device='cuda:0') tensor([[0, 3, 4, 5, 2, 1, 2, 4, 3, 5, 2, 3, 4, 5, 1, 3, 2, 4, 0, 1, 4, 3, 2, 1,
         0, 5, 2, 1, 0, 4],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
         4, 5, 5, 5, 5, 5]], device='cuda:0') '''
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            # print("batch_gated_edge_index", batch_gated_edge_index)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)

            gcn_outs.append(gcn_out)  # Graph Attention-Based Forecasting - Feature Extractor

        gcn_out = torch.cat(gcn_outs, dim=1)
        # print("GDN_gcn_out_1.shape", gcn_out.shape)    # torch.Size([224, 64])
        gcn_out = gcn_out.view(batch_num, node_num, -1)
        # print("GDN_gcn_out_2.shape", gcn_out.shape)    # torch.Size([32, 7, 64])

        indexes = torch.arange(0, node_num).to(device)
        # print(gcn_out)
        gcn_out = torch.mul(gcn_out, self.embedding(indexes))
        # print(gcn_out, self.embedding(indexes))  # tensor([0, 1, 2, 3, 4, 5, 6], device='cuda:0')
        '''
        gcn_out = gcn_out.permute(0, 2, 1)
        gcn_out = F.relu(self.bn_outlayer_in(gcn_out))
        gcn_out = gcn_out.permute(0, 2, 1)
        # print("GDN_gcn_out.shape", gcn_out.shape)    # torch.Size([32, 7, 64])
        '''
        # tcn_input = nn.AdaptiveMaxPool2d((gcn_out.shape[1], 1))(gcn_out)
        tcn_input = nn.AvgPool2d((gcn_out.shape[1], 1))(gcn_out)
        # tcn_input = F.adaptive_avg_pool2d(gcn_out, (1, gcn_out.shape[2]))
        # print(tcn_input)
        # print("tcn_input.shape", tcn_input.shape)  # torch.Size([32, 1, 64])
        tcn_input = nn.Softmax(2)(tcn_input)  # 64所在的维度参数和为1，在时间维度添加注意力
        # print(tcn_input)
        # print("tcn_input.shape", tcn_input.shape)  # torch.Size([32, 1, 64])
        # print(tcn_input.shape[1])
        feature = tcn_input.repeat(1, gcn_out.shape[1], 1)
        # print("feature.shape", feature.shape)  # torch.Size([32, 7, 64])

        L = torch.mul(gcn_out, feature)
        # print(L.shape)  # torch.Size([32, 7, 64]) 每一维数字变化不大基本都为0.0004，0.0005，导致反向传播参数变化也很小

        L = L.unsqueeze(1)
        # print(L.shape)  # torch.Size([32, 1, 7, 64])
        tcn_out = self.temp_layer(L)
        # print("GDN_tcn_out.shape:", tcn_out.shape)    # torch.Size([32, 1, 7, 64])
        out = tcn_out.squeeze()
        # print("tcn_out.shape:", out.shape)  # torch.Size([32, 7, 64])

        # out = self.dp(out)
        out = self.out_layer(out)  # Graph Attention-Based Forecasting - Out Layer
        # print("model_out", out.shape)    # torch.Size([32, 7, 1])
        out = out[:, 0, :]
        # print("model_out.shape:", out.shape)    # ([32, 1])

        ar_output = self.ar(x_raw)
        # print(ar_output.shape)  # torch.Size([32, 1])
        out = out + ar_output
        # out = out.view(-1, node_num)
        return out
