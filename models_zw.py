from layers import GraphConvolution
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
from GNNConv import GNN_node, GNN_node_Virtualnode
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

# 紫薇添加的
def gaussian_kernel_similarity(query, history_keys, sigma=1.0):
    """
    使用高斯核函数计算 query 和 history_keys 之间的相似度，并防止数值不稳定。

    参数:
        query (torch.Tensor): 查询向量，形状为 (1, d)。
        history_keys (torch.Tensor): 历史键向量，形状为 (seq-1, d)。
        sigma (float): 高斯核的宽度参数，默认值为 1.0。

    返回:
        torch.Tensor: 计算出的相似度权重，形状为 (1, seq-1)。
    """
    # 检查输入中是否有 NaN 或 Inf 值

    query = min_max_normalize(query)
    history_keys = min_max_normalize(history_keys)
    # 计算 query 和 history_keys 之间的欧氏距离
    distances = torch.cdist(query, history_keys)  # (1, seq-1)

    # 限制距离值，避免数值过大或过小
    distances = torch.clamp(distances, min=1e-8, max=1e8)

    # 使用高斯核函数计算相似度
    visit_weight = torch.exp(-distances ** 2 / (2 * sigma ** 2))  # (1, seq-1)

    # 对相似度进行归一化处理
    visit_weight = visit_weight / visit_weight.sum(dim=-1, keepdim=True)

    return visit_weight

class SimilarityCalculator(nn.Module):
    def __init__(self):
        super(SimilarityCalculator, self).__init__()

    def forward(self, o1, o2):
        # 计算点积相似度
        similarity = torch.bmm(o1, o2.transpose(1, 2))  # [1, 1, 1]
        return similarity


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    # pad 方法的主要作用是将一个矩阵列表填充为一个大矩阵，以便进行批量处理。
    # 具体来说，该方法将多个小矩阵拼接成一个大矩阵，并在必要位置填充指定的值（如0），使得每个小矩阵在大矩阵中占据连续的区域。
    # 这样处理后的大矩阵可以方便地用于批量计算，例如在深度学习模型中进行批量训练或推理。
    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        # 这段代码的功能是从输入 inputs 中解包三个变量：fingerprints、adjacencies 和 molecular_sizes。这些变量分别代表分子指纹、邻接矩阵和分子大小。
        fingerprints, adjacencies, molecular_sizes = inputs
        # 这行代码的功能是将 fingerprints 列表中的所有张量沿着第一个维度拼接成一个单一的张量。
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        # 获取指纹的嵌入向量
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        # 这个循环用于更新节点的隐藏状态：计算隐藏向量，更新隐藏向量，最后返回更新后的隐藏向量。

        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

class GNNGraph(torch.nn.Module):

    def __init__( # 神经网络类型：gin virtual_node：是否使用虚拟节点，默认为True
        self, num_layer=5, emb_dim=300,
            # 虚拟节点的作用包括：
            # 收集整个图的信息，帮助捕获全局结构特征。
            # 作为信息中转站，增强节点间消息传递的灵活性。
            # 缓解由于图结构变化导致的模型性能波动。
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        # 跳连（Jumping
        # Knowledge, JK）策略是一种用于图神经网络（GNN）的方法，旨在整合不同层的信息。具体来说，JK策略允许模型在多层之间选择如何融合节点表示。常见的JK策略有：
        # "last"：只使用最后一层的输出。这是默认选项。
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        # 调用super()创建一个父类的代理对象。
        # 通过该代理对象调用__init__方法，执行父类的初始化过程。
        super(GNNGraph, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GNN_node(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )

        # Pooling function to generate whole-graph embeddings
        # @zw:设置池化方法
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
            h_node = self.gnn_node(batched_data)

            h_graph = self.pool(h_node, batched_data.batch)
            return h_graph
            # return self.graph_pred_linear(h_graph)

class MAB(torch.nn.Module):
    def __init__(
        self, Qdim, Kdim, Vdim, number_heads,
        use_ln=False, *args, **kwargs
    ): # use_ln=False参数用于决定是否应用LayerNorm
        # 调用父类torch.nn.Module的构造函数。是什么意思？
        super(MAB, self).__init__(*args, **kwargs)
        self.Vdim = Vdim
        self.number_heads = number_heads

        # 检查 Vdim 是否可以被 number_heads 整除，这是因为每个头会分得一部分特征维度。
        assert self.Vdim % self.number_heads == 0, \
            'the dim of features should be divisible by number_heads'

        # 初始化四个线性变换层，用于转换输入张量到适当的维度。Qdense将Qdim转换为Vdim，Kdense和Vdense都将Kdim转换为Vdim，Odense保持输出在Vdim维度。
        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        # 定义一个线性层，用于最后输出的变换。
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        # 如果 use_ln 为 True，那么使用 LayerNorm，否则不使用。
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X, Y):
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        # 获取 batch_size（批次大小）和每个头分配的维度大小 dim_split，即将 Vdim 均分到多个头上。
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        # 将Q、K和V按照dim_split分割并重新组合，以便为每个注意力头提供独立的数据
        # K（键）用于与查询Q进行点积运算，生成注意力分数矩阵。这个过程是通过计算Q与每个K之间的相似性来完成的。
        # V（值）则是在得到注意力权重之后，用来加权求和以产生最终的输出。
        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        # 计算注意力权重矩阵Attn，然后应用softmax函数使其成为概率分布。
        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        # 用注意力权重加权求和V_split，并将结果与原始的Q_split相加。最后，重新组合成原来的形状。
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        # 根据use_ln的值决定是否应用层归一化，然后通过Odense线性层，再根据use_ln决定是否再次应用层归一化。
        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):

    def __init__( # 构造函数
        self, in_dim, out_dim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(SAB, self).__init__(*args, **kwargs)
        # 创建并设置self.net为MAB对象，用于多头自注意力机制。参数包括输入维度、输出维度及头部数量等。
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X):
        # 调用了 self.net 的 __call__ 方法，即执行了 MAB 的前向传播。
        return self.net(X, X)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return torch.mm(adj, torch.mm(x, self.weight))

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, node_features, device=torch.device('cpu')):
    # def __init__(self, voc_size, emb_dim, adj, node_features=None, device=torch.device('cpu')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))
        self.adj = torch.FloatTensor(adj).to(device)
        self.node_features = node_features
        if node_features is not None:
            self.node_features = torch.FloatTensor(node_features).to(device)
        else:
            self.node_features = torch.eye(voc_size).to(device)


        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

        # Define an attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1, batch_first=True)

    def forward(self):
        # Perform GCN operations
        node_embedding = self.gcn1(self.node_features, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)

        # Add an extra dimension for attention
        node_embedding = node_embedding.unsqueeze(1)  # (batch_size, 1, emb_dim) for attention

        # Apply attention mechanism
        attn_output, attn_output_weights = self.attention(node_embedding, node_embedding, node_embedding)

        # Remove the extra dimension
        node_embedding = attn_output.squeeze(1)  # (batch_size, emb_dim)

        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class FeatureExtractor(nn.Module):
    def __init__(self,  node_features_size,embedding_dim=131,device=torch.device('cpu:0')):
        super(FeatureExtractor, self).__init__()
        self.node_features_size = node_features_size
        self.device = device
        self.feature_embedding = nn.Embedding(node_features_size[0], embedding_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.encoder = nn.GRU(embedding_dim, embedding_dim, batch_first=True)

    def forward(self, node_features):
        features_list = []
        for feature in node_features:
            feature = self.dropout(self.feature_embedding(feature).to(self.device))
            feature = self.encoder(feature)
            features_list.append(feature)
        features_matrix = torch.stack(features_list, dim=0)
        return features_matrix

class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, MPNNSet, N_fingerprints, average_projection, node_features,substruct_num,substruct_para, substruct_dim=64,emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(GAMENet, self).__init__()
        K = len(vocab_size)+1
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4)

        self.transformerlayer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformerlayer, num_layers=1)
        self.linear_layer = nn.Linear(in_features=64, out_features=128)

        self.attn_linear = nn.Linear(emb_dim * 2,  emb_dim)
        self.attn_linear_1 = nn.Linear(emb_dim * 4,  emb_dim * 2)

        self.attention_layer = nn.MultiheadAttention(embed_dim=emb_dim * 2, num_heads=8)
        self.linear_layer_new = nn.Linear(emb_dim * 2, emb_dim)

        self.crosssAtte = nn.MultiheadAttention(emb_dim * 3 , 1, dropout=0.2, batch_first=True)
        # 药物分子层面建模
        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        self.MPNN_emb = MolecularGraphNeuralNetwork(
            N_fingerprints, emb_dim, layer_hidden=2, device=device
        ).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(
            average_projection.to(device=self.device),
            self.MPNN_emb.to(device=self.device),
        )
        self.MPNN_emb.to(device=self.device)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)  # @sub
        self.substruct_encoder = GNNGraph(**substruct_para)  # @sub

        # 原版
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj,node_features =node_features, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj,node_features =node_features, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.self_attention = nn.MultiheadAttention(embed_dim = emb_dim * 3, num_heads=8)
        self.similarity_calculator =SimilarityCalculator()
        self.init_weights()

    def forward(self, input, substruct_data):
        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)诊断
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))) # 手术
            i1_seq.append(i1) # 所有的诊断
            i2_seq.append(i2) # 所有的手术
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim) torch.Size([1, 1, 64])
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim) torch.Size([1, 1, 64])

        o1 = self.transformer_encoder(i1_seq)  # [1,1,64]
        o2 = self.transformer_encoder(i2_seq)  # [1,1,64]
        o1 = self.linear_layer(o1)  # torch.Size([1, 1, 128])
        o2 = self.linear_layer(o2)  # torch.Size([1, 1, 128])
        patient_representations = torch.cat([o1, o2],dim=-1).squeeze(dim=0)
        queries = self.query(patient_representations)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter
        else:
            drug_memory = self.ehr_gcn()

        "memory attention"
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match)).expand(131, 131).transpose(0, 1) # torch.Size([1, 131])
        drug_memory = torch.bmm(MPNN_att.unsqueeze(0), drug_memory.unsqueeze(0)).squeeze(0)

        "substructure_memory bank"
        substruct_weight = torch.sigmoid(self.substruct_rela(query)) # torch.Size([1, 491])
        substruct_embeddings = self.sab(self.substruct_encoder(**substruct_data).unsqueeze(0)).squeeze(0) # torch.Size([491, 64])
        substruct_weight_expanded = substruct_weight.unsqueeze(-1)  # 形状变为 [1, 491, 1]
        substruct_weight_expanded = substruct_weight_expanded.expand(-1, -1, 64)  # 形状变为 [1, 4
        weighted_sum =(substruct_weight_expanded * substruct_embeddings).sum(dim=1)
        weighted_sum = torch.nn.functional.normalize(weighted_sum, dim=-1) # [1,64]

        if len(input) > 1:
            # 记录这个输出的维度
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1

            history_values = torch.FloatTensor(history_values).to(self.device) # history_values:[1,131]

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # torch.Size([1, 64])
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)
        concatenated = torch.cat((fact1, weighted_sum), dim=1)
        attention_output, _ = self.attention_layer(concatenated, concatenated, concatenated)
        fact1 = self.linear_layer_new(attention_output)


        if len(input) > 1:

            visit_weight = gaussian_kernel_similarity(query, history_keys, sigma=0.5)
            visit_weight = visit_weight + F.sigmoid(visit_weight)

            weighted_values =visit_weight.mm(history_values)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim) 原版

        else:
            fact2 = fact1

        '''R:convert O and predict'''
        final = torch.cat([query, fact1, fact2], dim=-1)
        output,_ = self.crosssAtte(final,final,final)
        output = self.output(output * final) # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)







