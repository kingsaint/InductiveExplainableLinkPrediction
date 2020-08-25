from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import math
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUGE_INT = 1e31

class MultiheadAttention(nn.Module):
    def __init__(self, head_dim, num_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.Transform = nn.ModuleList()

        for _ in range(num_heads):
            self.Transform.append(nn.Sequential(nn.Dropout(dropout),
                                                nn.Linear(head_dim, 1),
                                                nn.LeakyReLU(0.1)))

    def forward(self, fact_embeddings, masks, layer_num):
        all_attn_weights = []
        for T in self.Transform:
            logits = T(fact_embeddings).squeeze(2)
            attn_weights = F.softmax(logits - HUGE_INT * (1 - masks), dim=1)
            all_attn_weights.append(attn_weights)
        return all_attn_weights


class GAT(nn.Module):
    def __init__(self, kg, entity_embed_dim, relation_embed_dim, num_layers, num_heads, head_dim, dropout):
        super(GAT, self).__init__()
        print(kg.num_entities)
        print(kg.num_relations)
        self.emb_e = nn.Embedding(kg.num_entities, entity_embed_dim, padding_idx=0)
        self.emb_r = nn.Embedding(kg.num_relations, relation_embed_dim, padding_idx=0)
        self.attentions = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(self.num_layers):
            self.attentions.append(MultiheadAttention(head_dim, num_heads, dropout))
        for i in range(self.num_layers):
            if i == 0:
                self.feed_forwards.append(nn.Linear(2*entity_embed_dim + relation_embed_dim, head_dim))
            else:
                self.feed_forwards.append(nn.Linear(num_heads*head_dim + entity_embed_dim + relation_embed_dim, head_dim))

    def initialize_modules(self):
        nn.init.xavier_normal_(self.emb_e.weight)
        nn.init.xavier_normal_(self.emb_r.weight)

    def forward(self, batch_e1, batch_q, dg, num_max_neigbors, mode):
        emb_e1 = self.emb_e(torch.LongTensor(batch_e1).to(device))
        emb_q = self.emb_r(torch.LongTensor(batch_q).to(device))

        h = emb_e1
        h_ = h
        if mode == 'train':
            neighbors = [dg.get_action_space(dg.training_graph, e1, q) for e1, q in zip(batch_e1, batch_q)]
        else:
            neighbors = [dg.get_action_space(dg.eval_graph, e1, q) for e1, q in zip(batch_e1, batch_q)]

        masks = []
        for i, n_i in enumerate(neighbors):
            if len(n_i) > num_max_neigbors:
                n_i = random.sample(n_i, num_max_neigbors)
                mask = [1.0 for j in range(len(n_i))]
                masks.append(mask)
                neighbors[i] = n_i
            else:
                mask = [1.0 for j in range(len(n_i))] + [0.0 for j in range(num_max_neigbors - len(n_i))]
                masks.append(mask)
                neighbors[i] += [[0, 0] for j in range(num_max_neigbors - len(n_i))] # Padding

        neighbors = torch.LongTensor(neighbors).to(device)
        r = self.emb_r(neighbors[:, :, 0])
        e = self.emb_e(neighbors[:, :, 1])

        masks = torch.FloatTensor(masks).to(device)
        masks.requires_grad = True

        layer_num = 0
        for attention, feed_forward in zip(self.attentions, self.feed_forwards):
            h_ = h_.unsqueeze(1).expand(-1, num_max_neigbors, -1)
            fact_embeddings = torch.cat([h_, r, e], dim=2)

            # A Linear transformation of the fact embeddings
            x = feed_forward(fact_embeddings)
            A = attention(x, masks, layer_num) # Multihead attention
            A = [a.unsqueeze(2) for a in A]
            if layer_num == self.num_layers - 1:
                x = torch.cat([torch.bmm(torch.transpose(a, 2, 1), x) for a in A], dim=1)
                x = torch.mean(x, dim=1)
                h = h + x # Residual connection
            else:
                x = torch.cat([F.leaky_relu(torch.bmm(torch.transpose(a, 2, 1), x)) for a in A], dim=2).squeeze(1)
                h_ = x
            layer_num += 1
        #print(h)
        #print(emb_q)
        return h

#from src.directed_graph import DirectedGraph
#dg = DirectedGraph('/ExplainableKGReasoning/data/umls')
#print(dg.training_graph)
#print(dg.get_action_space(dg.training_graph, 15, 6))

#gat = GAT(kg=dg, entity_embed_dim=200, relation_embed_dim=200, num_layers=2, num_heads=4, head_dim=200, dropout=0.3)
#h = gat([15, 18], [6, 12], dg, 400)
#print(h.size())


