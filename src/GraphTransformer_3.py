from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from torch.distributions.bernoulli import Bernoulli
import math
import random
import time
device = torch.device("cuda" if cuda.is_available() else "cpu")
print(device)
HUGE_INT = 1e31

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.Query = nn.ModuleList()
        self.Key = nn.ModuleList()
        self.Value = nn.ModuleList()
        self.head_dim = head_dim
        assert  head_dim * num_heads == embed_dim

        for _ in range(num_heads):
            self.Query.append(nn.Linear(embed_dim, head_dim))
            self.Key.append(nn.Linear(embed_dim, head_dim))
            self.Value.append(nn.Linear(embed_dim, head_dim))

    def forward(self, query, key, value, masks):
        attn_output = []
        for Q, K, V in zip(self.Query, self.Key, self.Value):
            #print(Q(query).size())
            #print(K(key).size())
            #print(torch.transpose(K(key), 2, 1).size())
            #print(V(value).size())

            attn_weights = F.softmax((torch.bmm(Q(query), torch.transpose(K(key), 2, 1)) / math.sqrt(self.head_dim)) - HUGE_INT * (1 - masks.unsqueeze(1)), dim=2)
            #print(attn_weights.size())
            #print(masks.size())
            attn_output.append(torch.bmm(attn_weights, V(value)))
        attn_output = torch.cat(attn_output, dim=2)
        #print(attn_output.size())
        return attn_output


class GraphTransformer(nn.Module):
    def __init__(self, kg, num_layers, num_heads, dropout, embed_dim, hidden_dim, neighbor_dropout_rate):
        super(GraphTransformer, self).__init__()
        self.emb_e = nn.Embedding(kg.num_entities, embed_dim, padding_idx=0)
        self.emb_r = nn.Embedding(kg.num_relations, embed_dim, padding_idx=0)
        self.msg_enc = nn.Sequential(nn.Linear(3*embed_dim, embed_dim),
                                nn.LeakyReLU()
                                )
        self.dropout = nn.Dropout(dropout)
        self.bernoulli_dist = Bernoulli(torch.FloatTensor([1 - neighbor_dropout_rate]))

        self.attentions = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        self.layernorm_1 = nn.ModuleList()
        self.layernorm_2 = nn.ModuleList()

        head_dim = embed_dim // num_heads
        #print(head_dim)
        #print(num_heads)
        for _ in range(num_layers):
            self.attentions.append(MultiheadAttention(embed_dim, head_dim, num_heads, dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)
                                                    ))
            self.layernorm_1.append(nn.LayerNorm(embed_dim, eps=1e-05))
            self.layernorm_2.append(nn.LayerNorm(embed_dim, eps=1e-05))


    def initialize_modules(self):
        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_normal_(self.emb_r.weight)

    def get_neighbors(self, graph, e1, q):
        # print(e1, q)
        action_space = [[r, e2] for (r, e2) in graph[e1] if r != q and r != q + 1]
        # print(action_space)
        return action_space

    def vectorize_neighbors(self, batch_e1, batch_q, graph, num_max_neighbors, mode):
        neighbors = [self.get_neighbors(graph, e1, q) for e1, q in zip(batch_e1, batch_q)]

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
        with torch.no_grad():
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

    def forward(self, batch_e1, batch_q, graph, seen_entities, num_max_neighbors, mode):
        #print(batch_e1)
        #print(batch_q)
        #print(seen_entities)
        #print(mode)

        if mode == 'test':
            batch_e1_aug = batch_e1.clone() # Changes made here
            for i in range(batch_e1_aug.size()[0]):
                #print(batch_e1_aug[i].item())
                if batch_e1_aug[i].item() not in seen_entities:
                    batch_e1_aug[i] = 0
            #print(batch_e1_aug)
            emb_e1 = self.emb_e(batch_e1_aug) #torch.LongTensor(batch_e1).to(device)
            #time.sleep(30)
        else:
            emb_e1 = self.emb_e(batch_e1)
        emb_q = self.emb_r(batch_q) #torch.LongTensor(batch_q).to(device)
        #emb_e2 = self.emb_e(torch.LongTensor(batch_e2).to(device))
        #print(emb_e1)
        h = emb_e1
        h_ = h.unsqueeze(1).expand(-1, num_max_neighbors, -1)
        #print(h_.size())
        #print(batch_e1)
        (r, e), masks = self.vectorize_neighbors(batch_e1.cpu().numpy().tolist(), batch_q.cpu().numpy().tolist(), graph, num_max_neighbors, mode)

        #print("Masks =", masks.size())
        #key = torch.cat([h_, r, e], dim=2)
        key = r
        value = self.msg_enc(torch.cat([h_, r, e], dim=2))
        #print("Value = ", value.size())
        query = emb_q.unsqueeze(1)
        #print("Query = ", query.size())

        for attention, ln_1, feed_forward, ln_2 in zip(self.attentions, self.layernorm_1, self.feed_forwards, self.layernorm_2):

            x = attention(query, key, value, masks) # Multihead attention
            x = x.squeeze(1)
            #print(x.size())
            x = self.dropout(x)  # Dropout
            h = h + x  # Residual connection
            #print(h.size())
            h = ln_1(h) # layer norm

            x = feed_forward(h) # feed forward
            x = self.dropout(x) # Dropout
            h = h + x # Residual connection
            h = ln_2(h) # layer norm
            #print("------------------------------------------------")

            h_ = h.unsqueeze(1).expand(-1, num_max_neighbors, -1)
            #key = torch.cat([h_, r, e], dim=2)
            value = self.msg_enc(torch.cat([h_, r, e], dim=2))

        #dissimilarity = torch.norm(h + emb_q - emb_e2, p=2, dim=1)
        #return dissimilarity
        return h, emb_q #, emb_e2


