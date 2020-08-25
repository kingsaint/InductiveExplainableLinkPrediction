"""
Copyright (c), 2020, Rajarshi Bhowmik
All rights reserved
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

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

import pdb

class NoGraphTransformer(nn.Module):
    def __init__(self, kg, dropout, embed_dim):
        super(NoGraphTransformer, self).__init__()
        self.emb_e = nn.Embedding(kg.num_entities, embed_dim, padding_idx=0)
        self.emb_r = nn.Embedding(kg.num_relations, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_normal_(self.emb_r.weight)


    def forward(self, batch_e1, batch_q, graph, seen_entities, num_max_neighbors, mode):
        print(mode)
        if mode == 'test':
            batch_e1_aug = batch_e1.clone()  # Changes made here
            for i in range(batch_e1_aug.size()[0]):
                if batch_e1_aug[i].item() not in seen_entities:
                    batch_e1_aug[i] = 0
            emb_e1 = self.emb_e(batch_e1_aug)
        else:
            emb_e1 = self.emb_e(batch_e1)
        pdb.set_trace()
        emb_q = self.emb_r(batch_q)

        h = emb_e1

        return h, emb_q


