import os
from directed_graph import DirectedGraph
from queue import Queue
import copy
import sys
import numpy as np

sys.setrecursionlimit(10000)

# class BFS:
#     def __init__(self, graph):
#         self.queue = []
#         self.queue_dict = {}
#         self.visited_dict = {}
#         self.paths_to_target = []
#         self.graph = graph
#
#     def search(self, rel, target):
#         vertex = self.queue[0]
#         # print(vertex)
#         (r, e), depth, path_to_me = vertex
#         del self.queue[0]
#         del self.queue_dict[(r, e)]
#         self.visited_dict[(r, e)] = True
#
#         #print("visited ", (r, e), "depth=", depth, "path to the node = ", path_to_me)
#         # print(visited)
#         if depth >= 3: # retrict the depth of search
#             # print(path_to_me)
#             return
#         else:
#             if path_to_me[-1][-1] == target:
#                 self.paths_to_target.append(path_to_me)
#
#         children = [(r_, e_) for (r_, e_) in self.graph[e] if r_ != rel and r_ != rel + 1]
#         for c in children:
#             if c not in self.queue_dict and c not in self.visited_dict:
#                 # print(c)
#                 p = copy.deepcopy(path_to_me)
#                 p.append(c)
#                 # print(p)
#                 self.queue.append((c, depth + 1, p))
#                 self.queue_dict[c] = True
#         # print(queue)
#         if len(self.queue) > 0:
#             self.search(rel, target)
#         else:
#             return
#
#
# data_dir = '../data/WN18RR'
#
# dg = DirectedGraph(data_dir)
#
# print(dg.rel2id)
# print(len(dg.training_graph))
#
# #graph = {1:[(1, 2), (2, 3), (3, 5)], 2: [(2, 4), (2, 5)], 3: [(1, 2), (3, 4)], 4: [], 5:[(3, 2)]}
#
# rel_cooccurrence_matrix = [[0 for i in range(len(dg.rel2id))] for j in range(len(dg.rel2id))]
# rel_cooccurrence_matrix = np.array(rel_cooccurrence_matrix)
# print(rel_cooccurrence_matrix.shape)
#
# with open(os.path.join(data_dir, 'train.triples')) as f:
#     for triple in f:
#         h, t, q = triple.strip().split('\t')
#         h = dg.entity2id[h]
#         q = dg.rel2id[q]
#         t = dg.entity2id[t]
#         print(h, q, t)
#
#         # Initialization
#
#         queue = []
#         queue_dict = {}
#         bfs = BFS(dg.training_graph)
#         path_to_me = [(0, h)]
#         bfs.queue.append(((0, h), 0, path_to_me))
#         bfs.queue_dict[(0, h)] = True
#         rel = q
#         target = t
#
#
#         bfs.search(rel, target)
#         paths_to_target = bfs.paths_to_target
#         print(paths_to_target)
#
#         for path in paths_to_target:
#             for i in range(1, len(path)):
#                 r, _ = path[i]
#                 # print(r)
#                 rel_cooccurrence_matrix[q][r] += 1
#
# np.save('WN18RR.npy', rel_cooccurrence_matrix)



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

data_dir = '../data/umls'

dg = DirectedGraph(data_dir)



x = list(dg.rel2id.keys())
y = list(dg.rel2id.keys())

rel_cooccurrence_matrix = np.load('umls.npy')

fig, ax = plt.subplots()
im = ax.imshow(rel_cooccurrence_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(y)))
ax.set_yticks(np.arange(len(x)))
# ... and label them with the respective list entries
ax.set_xticklabels(y)
ax.set_yticklabels(x)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# for i in range(len(x)):
#     for j in range(len(y)):
#         text = ax.text(j, i, rel_cooccurrence_matrix[i, j],
#                        ha="center", va="center", color="w")

ax.set_title("xx")
fig.tight_layout()
plt.show()




