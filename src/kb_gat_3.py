from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import math
import random
import numpy as np

# Sunsampling of neighborhood, embedding dropouts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUGE_INT = 1e31

class MultiheadAttention(nn.Module):
    def __init__(self, entity_dim, relation_dim, head_dim, num_heads, dropout, l):
        super(MultiheadAttention, self).__init__()
        self.Transform = nn.ModuleList()
        self.FactProjector = nn.ModuleList()

        if l == 0:
            for _ in range(num_heads):
                self.FactProjector.append(nn.Linear(2*entity_dim + relation_dim, head_dim))
                self.Transform.append(nn.Sequential(
                                                    nn.Dropout(dropout),
                                                    nn.Linear(head_dim, relation_dim),
                                                    nn.LeakyReLU(0.1)
                                                    ))
        else:
            for _ in range(num_heads):
                self.FactProjector.append(nn.Linear(num_heads*head_dim + entity_dim + relation_dim, head_dim),)
                self.Transform.append(nn.Sequential(nn.Dropout(dropout),
                                                    nn.Linear(head_dim, relation_dim),
                                                    nn.LeakyReLU(0.1)
                                                    ))

    def forward(self, fact_embeddings, query_embeddings, masks, layer_num):
        A = []
        H = []
        for FP, T in zip(self.FactProjector, self.Transform):
            x = T(FP(fact_embeddings))
            #print(x.size())
            logits = torch.sum(x * query_embeddings, dim=2)
            attn_weights = F.softmax(logits - HUGE_INT * (1 - masks), dim=1).unsqueeze(2)
            #print(attn_weights.size())
            x = F.leaky_relu(torch.bmm(torch.transpose(attn_weights, 2, 1), x))
            #print(x.size())
            H.append(x.squeeze(1))
            A.append(attn_weights.squeeze(2))
        return A, H


class GAT(nn.Module):
    def __init__(self, kg, entity_dim, relation_dim, num_layers, num_heads, head_dim, dropout, neighbor_dropout_rate):
        super(GAT, self).__init__()
        #print(kg.num_entities)
        #print(kg.num_relations)
        self.emb_e = nn.Embedding(kg.num_entities, entity_dim, padding_idx=0)
        self.emb_r = nn.Embedding(kg.num_relations, relation_dim, padding_idx=0)
        self.attentions = nn.ModuleList()
        self.query_projector = nn.Linear(relation_dim, relation_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.entity_dim = entity_dim
        self.bernoulli_dist = Bernoulli(torch.FloatTensor([1 -  neighbor_dropout_rate]))

        for l in range(self.num_layers):
            self.attentions.append(MultiheadAttention(entity_dim, relation_dim, head_dim, num_heads, dropout, l))


    def initialize_modules(self):
        nn.init.xavier_normal_(self.emb_e.weight)
        nn.init.xavier_normal_(self.emb_r.weight)

    def vectorize_action_space(self, batch_e1, batch_q, dg, num_max_neighbors, mode):
        if mode == 'train':
            neighbors = [dg.get_action_space(dg.training_graph, e1, q) for e1, q in zip(batch_e1, batch_q)]
        else:
            neighbors = [dg.get_action_space(dg.eval_graph, e1, q) for e1, q in zip(batch_e1, batch_q)]

        masks = []
        for i, n_i in enumerate(neighbors):
            if len(n_i) > num_max_neighbors:
                n_i = random.sample(n_i, num_max_neighbors)
                mask = [1.0 for j in range(len(n_i))]
                masks.append(mask)
                neighbors[i] = n_i
            else:
                mask = [1.0 for j in range(len(n_i))] + [0.0 for j in range(num_max_neighbors - len(n_i))]
                masks.append(mask)
                neighbors[i] += [[0, 0] for j in range(num_max_neighbors - len(n_i))] # Padding

        neighbors = torch.LongTensor(neighbors).to(device)
        r = self.dropout(self.emb_r(neighbors[:, :, 0]))
        e = self.dropout(self.emb_e(neighbors[:, :, 1]))

        masks = torch.FloatTensor(masks).to(device)
        if mode == 'train':
            neighbor_dropout = self.bernoulli_dist.sample([len(batch_e1), num_max_neighbors]).squeeze(2).to(device)
            #print(neighbor_dropout)
            masks = masks * neighbor_dropout
        masks.requires_grad = True
        #print(masks.size())

        return (r, e), masks

    def forward(self, batch_e1, batch_q, dg, num_max_neighbors, mode):

        emb_e1 = self.emb_e(batch_e1) #torch.LongTensor(batch_e1).to(device)
        emb_q = self.emb_r(batch_q) #torch.LongTensor(batch_q).to(device)

        h = self.dropout(emb_e1)
        h_ = h
        (r, e), masks = self.vectorize_action_space(batch_e1, batch_q, dg, num_max_neighbors, mode)
        query_embeddings = self.query_projector(self.dropout(emb_q)).unsqueeze(1).expand(-1, num_max_neighbors, -1)

        for layer_num, attention in enumerate(self.attentions):
            h_ = h_.unsqueeze(1).expand(-1, num_max_neighbors, -1)
            fact_embeddings = torch.cat([h_, r, e], dim=2)

            # A Linear transformation of the fact embeddings
            A, H = attention(fact_embeddings, query_embeddings, masks, layer_num) # Multihead attention
            if layer_num == self.num_layers - 1:
                #print(H)
                X = torch.cat(H, dim=1).view(-1, self.num_heads, self.entity_dim)
                #print(X.size())
                X = torch.mean(X, dim=1)
                X = F.leaky_relu(X)
                #print(X.size())
                h = h + X # Residual connection
            else:
                X = torch.cat(H, dim=1)
                #print(X.size())
                h_ = X
            #print("----------------------")
            #layer_num += 1
        #print(h)
        #print(emb_q)
        return h, emb_q

#from src.directed_graph import DirectedGraph
#dg = DirectedGraph('/ExplainableKGReasoning/data/umls')
#print(dg.training_graph)
#print(dg.get_action_space(dg.training_graph, 15, 6))

#gat = GAT(kg=dg, entity_dim=200, relation_dim=200, num_layers=2, num_heads=4, head_dim=200, dropout=0.3, neighbor_dropout_rate=0.5)
#h = gat([15, 18], [6, 12], dg, 400,  mode='train')
#print(h.size())


