from networkx.algorithms import matching
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
import torch_geometric.utils as ut
import utils.backbone
from BB_GM.affinity_layer import InnerProductWithWeightsAffinity
from BB_GM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from torch_geometric.nn import GCNConv

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Permuter(torch.nn.Module):
    def __init__(self, node_hidden_dim):
        super().__init__()
        self.scoring_fc = nn.Linear(node_hidden_dim, 1)

    def score(self, x, mask):
        scores = self.scoring_fc(x)
        fill_value = scores.min().item() - 1
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
        return scores

    def soft_sort(self, scores, hard, tau):
        scores_sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - scores_sorted).abs().neg() / tau
        perm = pairwise_diff.softmax(-1)
        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
        return perm

    def mask_perm(self, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    def forward(self, node_features, mask, hard=False, tau=1.0):
        # add noise to break symmetry
        node_features = node_features + torch.randn_like(node_features) * 0.05
        scores = self.score(node_features, mask)
        perm = self.soft_sort(scores, hard, tau)
        perm = perm.transpose(2, 1)
        perm = self.mask_perm(perm, mask)
        return perm

    @staticmethod
    def permute_node_features(node_features, perm):
        node_features = torch.matmul(perm, node_features)
        return node_features

    @staticmethod
    def permute_edge_features(edge_features, perm):
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features)
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features.permute(0, 2, 1, 3))
        edge_features = edge_features.permute(0, 2, 1, 3)
        return edge_features

    @staticmethod
    def permute_graph(graph, perm):
        graph.node_features = Permuter.permute_node_features(graph.node_features, perm)
        graph.edge_features = Permuter.permute_edge_features(graph.edge_features, perm)
        return graph


class Net(utils.backbone.VGG16_bn):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_dim = 8
        self.mpnn = GCNConv(in_channels=4, out_channels=self.hidden_dim)
          
        self.permuter = Permuter(self.hidden_dim)



    def forward(
        self,
        images,
        graphs,
        n_points,
        perm_mats,
        mask,
        visualize_flag=False,
        visualization_params=None,
    ):
        '''
        1. 使用MPNN更新Gs 和 Gt
        2. 从Gs中选择top-k个点构成集合Ps,从Gt中选择top-k个点构成集合Pt.
            1.选拔机制怎么定义, nn.linear
                
        3. 对Ps和Pt进行匹配,反馈gradients

        '''
        global_list = []
        orig_graph_list = []
        topk_list = []
        mask_list = []
        for graph, n_point in zip(graphs, n_points):

            node_feat = graph.x
            edge_index = graph.edge_index
            
            node_feat = self.mpnn(node_feat, edge_index)
            # node_feat = F.relu(node_feat)
            
            node_feat_dense, batch_mask = ut.to_dense_batch(node_feat, graph.batch)
            mask_list.append(batch_mask)

            mask = graph.mask
            mask = ut.to_dense_batch(mask, graph.batch)[0]

            perm = self.permuter(node_feat_dense, mask, hard=True, tau=1.0)
            print('perm size: ', perm.size())
            print('dense_node_feat size: ', node_feat_dense.size())

            #  perm: b*n*n dense_node_feat: b*n*d 
            new_node_feat = torch.bmm(perm, node_feat_dense)[:,:5,:]

            topk_list.append(new_node_feat)

        
        # 对topk_list中的source batch 和 target batch进行匹配
        # source batch: b*n*d * mask
        # print(topk_list[0].size(), topk_list[1].size())
        
        for sg,tg in zip(topk_list[0], topk_list[1]):
            print(sg.size(), tg.size())
            break

        matchings = None
        
        return matchings
