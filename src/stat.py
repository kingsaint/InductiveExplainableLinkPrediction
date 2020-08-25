import os

data_dir = '../data'
data_set = 'NELL-995.test'

test_graph = {}

with open(os.path.join(data_dir, data_set, 'test.triples')) as f:
    for triple in f:
        e1, e2, r = triple.strip().split('\t')
        if e1 not in test_graph:
            test_graph[e1] = []
        if (r, e2) not in test_graph[e1]:
            test_graph[e1].append((r, e2))

with open(os.path.join(data_dir, data_set, 'aux.triples')) as f:
    for triple in f:
        e1, e2, r = triple.strip().split()
        r_inv = r + '_inv'
        if e1 in test_graph:
            if (r, e2) not in test_graph[e1]:
                test_graph[e1].append((r, e2))
        if e2 in test_graph:
            if (r_inv, e1) not in test_graph[e2]:
                test_graph[e2].append((r_inv, e1))

count = 0
max_outdegree = 0
for e in test_graph:
    if len(test_graph[e]) == 1:
        print(e, test_graph[e])
        count += 1
    if len(test_graph[e]) > max_outdegree:
        max_outdegree = len(test_graph[e])




print(count)
print(len(test_graph))
print(max_outdegree)



