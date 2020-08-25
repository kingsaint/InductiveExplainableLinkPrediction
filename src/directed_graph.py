"""
Copyright (c), 2020, Rajarshi Bhowmik
All rights reserved
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

class Concept2Index():
    def __init__(self, data_dir):
        self.entity2id = {}
        self.rel2id = {}
        self.id2entity = {}
        self.id2rel = {}
        self.num_entities = 0
        self.num_relations = 0

        def load_index(input_path):

            index, rev_index = {}, {}
            with open(input_path, encoding='utf-8') as f:

                for i, line in enumerate(f.readlines()):

                    v, _ = line.strip().split('\t')
                    #print(v, i)
                    index[v] = i
                    rev_index[i] = v
            return index, rev_index

        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        self.rel2id, self.id2rel = load_index(os.path.join(data_dir, 'relation2id.txt'))
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.rel2id)


class UndirectedGraph(Concept2Index):
    def __init__(self, data_dir):
        Concept2Index.__init__(self, data_dir)
        self.seen_entity2id = self.entity2id.copy()
        self.seen_id2entity = self.id2entity.copy()
        self.training_graph = {}
        self.eval_graph = {}
        self.data_files = [os.path.join(data_dir, 'train.triples'), os.path.join(data_dir, 'dev.triples')] # Changes made here os.path.join(data_dir, 'test.triples'),
        if 'NELL' in data_dir:
            if 'test' in data_dir:
                self.data_files = [os.path.join(data_dir, 'train.triples'), os.path.join(data_dir, 'train.dev.large.triples'), os.path.join(data_dir, 'dev.triples')]
            else:
                self.data_files = [os.path.join(data_dir, 'train.triples'), os.path.join(data_dir, 'train.large.triples'), os.path.join(data_dir, 'dev.triples')]

        for data_file in self.data_files:
            with open(data_file, encoding='utf-8') as f:
                self.data = f.readlines()
                for triple in self.data:
                    triple = triple.replace("\r", '').replace('\n', '')
                    e1, e2, r = triple.split('\t')
                    r_inv = r + '_inv'
                    e1_id = self.entity2id[e1]
                    e2_id = self.entity2id[e2]
                    r_id = self.rel2id[r]
                    r_inv_id = self.rel2id[r_inv]

                    # Build the undirected KG
                    if e1_id not in self.eval_graph:
                        self.eval_graph[e1_id] = []
                    self.eval_graph[e1_id].append([r_id, e2_id])
                    self.eval_graph[e1_id].append([r_inv_id, e2_id])

                    if e2_id not in self.eval_graph:
                        self.eval_graph[e2_id] = []
                    self.eval_graph[e2_id].append([r_id, e1_id])
                    self.eval_graph[e2_id].append([r_inv_id, e1_id])

                    # Build the training subgraph of undirected KG
                    if 'train' in data_file:
                        if e1_id not in self.training_graph:
                            self.training_graph[e1_id] = []
                        self.training_graph[e1_id].append([r_id, e2_id])
                        self.training_graph[e1_id].append([r_inv_id, e2_id])

                        if e2_id not in self.training_graph:
                            self.training_graph[e2_id] = []
                        self.training_graph[e2_id].append([r_id, e1_id])
                        self.training_graph[e2_id].append([r_inv_id, e1_id])

        self.eval_graph[0] = [[2, 0]]
        self.training_graph[0] = [[2, 0]]


class DirectedGraph(Concept2Index):
    def __init__(self, data_dir):
        Concept2Index.__init__(self, data_dir)
        self.seen_entity2id = self.entity2id.copy()
        self.seen_id2entity = self.id2entity.copy()
        self.training_graph = {}
        self.eval_graph = {}
        self.aux_graph = {}

        self.data_files = [os.path.join(data_dir, 'train.triples'), os.path.join(data_dir, 'dev.triples')] # Changes made here os.path.join(data_dir, 'test.triples'),
        if 'NELL' in data_dir:
            if 'test' in data_dir:
                self.data_files = [os.path.join(data_dir, 'train.dev.triples'), os.path.join(data_dir, 'train.dev.large.triples'),  os.path.join(data_dir, 'dev.triples')]
            else:
                self.data_files = [os.path.join(data_dir, 'train.triples'), os.path.join(data_dir, 'train.large.triples'), os.path.join(data_dir, 'dev.triples')]

        for data_file in self.data_files:
            with open(data_file, encoding='utf-8') as f:
                self.data = f.readlines()
                for triple in self.data:
                    triple = triple.replace("\r", '').replace('\n', '')
                    e1, e2, r = triple.split('\t')
                    r_inv = r + '_inv'
                    e1_id = self.entity2id[e1]
                    e2_id = self.entity2id[e2]
                    r_id = self.rel2id[r]
                    r_inv_id = self.rel2id[r_inv]

                    # Build the directed KG
                    if e1_id not in self.eval_graph:
                        self.eval_graph[e1_id] = []
                    self.eval_graph[e1_id].append([r_id, e2_id])

                    if e2_id not in self.eval_graph:
                        self.eval_graph[e2_id] = []
                    self.eval_graph[e2_id].append([r_inv_id, e1_id])

                    # Build the training subgraph
                    if 'train' in data_file:
                        if e1_id not in self.training_graph:
                            self.training_graph[e1_id] = []
                        self.training_graph[e1_id].append([r_id, e2_id])

                        if e2_id not in self.training_graph:
                            self.training_graph[e2_id] = []
                        self.training_graph[e2_id].append([r_inv_id, e1_id])

        self.eval_graph[0] = [[2, 0]]
        self.training_graph[0] = [[2, 0]]

        self.load_aux_graph(data_dir) # Changes made here

    def load_aux_graph(self, data_dir): # Changes made here
        self.aux_graph.update(self.eval_graph)
        data_files = ['aux.triples', 'test.triples']
        for file in data_files:
            with open(os.path.join(data_dir, file), encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    e1, e2, r = line.strip().split()

                    if e1 not in self.entity2id:
                        e1_id = len(self.entity2id)
                        self.entity2id[e1] = e1_id
                        self.id2entity[e1_id] = e1

                    if e2 not in self.entity2id:
                        e2_id = len(self.entity2id)
                        self.entity2id[e2] = e2_id
                        self.id2entity[e2_id] = e2

                    if e1 in self.seen_entity2id and e2 not in self.seen_entity2id:
                        e1_id = self.entity2id[e1]
                        r_inv_id = self.rel2id[r + '_inv']
                        e2_id = self.entity2id[e2]
                        if e2_id not in self.aux_graph:
                            self.aux_graph[e2_id] = []
                        if [r_inv_id, e1_id] not in self.aux_graph[e2_id]:
                            self.aux_graph[e2_id].append([r_inv_id, e1_id])

                    if e2 in self.seen_entity2id and e1 not in self.seen_entity2id:
                        e1_id = self.entity2id[e1]
                        r_id = self.rel2id[r]
                        e2_id = self.entity2id[e2]
                        if e1_id not in self.aux_graph:
                            self.aux_graph[e1_id] = []
                        if [r_id, e2_id] not in self.aux_graph[e1_id]:
                            self.aux_graph[e1_id].append([r_id, e2_id])

    # A special case to deal with the scenarios where both entities are unknown (DEPRICATED)
#                    if e1 not in self.seen_entity2id and e2 not in self.seen_entity2id:
#                        e1_id = self.entity2id[e1]
#                        r_id = self.rel2id[r]
#                        r_inv_id = self.rel2id[r + '_inv']
#                        e2_id = self.entity2id[e2]
#                        if e1_id not in self.aux_graph:
#                            self.aux_graph[e1_id] = []
#                        if [r_id, e2_id] not in self.aux_graph[e1_id]:
#                            self.aux_graph[e1_id].append([r_id, e2_id])
#
#                        if e2_id not in self.aux_graph:
#                            self.aux_graph[e2_id] = []
#                        if [r_inv_id, e1_id] not in self.aux_graph[e2_id]:
#                            self.aux_graph[e2_id].append([r_inv_id, e1_id])

    def get_action_space(self, graph, e1, q):
        action_space = [[r, e2] for (r, e2) in graph[e1] if r != q and r != q + 1]
        return action_space


