import math
import torch
from torch import nn
# from labml_helpers.module import Module
from torch.nn.parameter import Parameter

# class GraphAttentionV2Layer(Module):
#     def __init__(self, in_features: int, out_features: int, n_heads: int,
#             is_concat: bool = True,
#             dropout: float = 0.6,
#             leaky_relu_negative_slope: float = 0.2,
#             share_weights: bool = False):
#         super().__init__()
#         self.is_concat = is_concat
#         self.n_heads = n_heads
#         self.share_weights = share_weights
#         if is_concat:
#             assert out_features % n_heads == 0
#             self.n_hidden = out_features // n_heads
#         else:
#             self.n_hidden = out_features
#         self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
#         if share_weights:
#             self.linear_r = self.linear_l
#         else:
#             self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
#         self.attn = nn.Linear(self.n_hidden, 1, bias=False)
#         self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
#         self.softmax = nn.Softmax(dim=1)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
#         # print(h.size())
#         n_nodes = h.shape[0]
#         g_l = self.linear_l(h).reshape(n_nodes, self.n_heads, self.n_hidden)
#         g_r = self.linear_r(h).reshape(n_nodes, self.n_heads, self.n_hidden)
#         g_l_repeat = g_l.repeat(n_nodes, 1, 1)
#         g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
#         g_sum = g_l_repeat + g_r_repeat_interleave
#         g_sum = g_sum.reshape(n_nodes, n_nodes, self.n_heads, self.n_hidden)
#         e = self.attn(self.activation(g_sum))
#         e = e.squeeze(-1)
#         e = e[...,0].squeeze()
#         assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
#         assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
#         # assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
#         e = e.masked_fill(adj_mat == 0, float('-inf'))
#         a = self.softmax(e)
#         a = self.dropout(a)
#         attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)
#         if self.is_concat:
#             return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
#         else:
#             return attn_res.mean(dim=1)



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'