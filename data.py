# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:26:09 2016

@author: yash

TODO:

[1] : design a checking methodology to evaliate sampling algorithms
"""

from __future__ import print_function
import time
import pickle
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx_viewer import Viewer
import random
import itertools

path = '/home/yash/Project/dataset/GraphSimilarity/'
#dataset = ['mutag.graph', 'ptc.graph', 'enzymes.graph', 'proteins.graph', 'nci1.graph', 'nci109.graph', 'collab.graph', 'imdb_action_romance.graph', 'reddit_iama_askreddit_atheism_trollx.graph', 'reddit_multi_5K.graph', 'reddit_subreddit_10K.graph']
dataset = ['reddit_multi_5K.graph']#'imdb_comedy_romance_scifi.graph']


#        
#class Data(object):
#    def __init__(self, path):
#        ##something
#        self.has_more = True
#        self.read_data(path)
#        
#    def read_data(path):
#        
#    def next_batch(self):
#        ##something    

def load_data(ds_name):
    f = open(ds_name, "rb")
    data = pickle.load(f, encoding='latin1')
    graph_data = data["graph"]
    labels = data["labels"]
    labels  = np.array(labels, dtype = np.float)
    return graph_data, labels


def create_adj_list(graph):
    print("Started creating adj_list...")
    adj_list = {}
    for k, v in graph.items():
        adj_list[k] = v['neighbors']
        
    print("Done")
    return adj_list

def sampling(G, size, **args):   
    node_count = len(G.keys())
        
    if size > node_count:
        #add dummy nodes
        for i in range(node_count, size):
            G[i] = {'neighbors':[]}
    
    elif size <  node_count:        
        selected = random_walk(G, size, args)
        G = subgraph(G, selected)
        
    #assign values to each node
    # a) default, 1
    # b) based on label
    # c) normalised degree
    for v in G.keys():
        G[v]['val'] = [1] #default
        
    return G


def subgraph(G, selected):
    G_new = {}
    for node, _ in selected:
        neighbors = [v for v in G[node]['neighbors'] if selected.get(v, 0)!=0]
        G_new[node] = {'neighbors':neighbors}        
    return G_new
    
    
def random_walk(G, size, seeds=20, start_node=None,  metropolized=True, **args):
    #TODO: disconnectednss can be a problem here, visualise the generate graph to fine tune algo
       
    if start_node==None:
        start_node = np.random.choice(list(G.keys()), seeds)
        
    v = start_node
    flag = True
    selected = {}
    
    while flag:
        for i in range(seeds):
            if metropolized:    # Metropolis Hastings Random Walk (with the uniform target node distribution) 
                candidate = np.random.choice(G[v[i]]['neighbors'])
                #v[i] = candidate if (np.random.rand() < len(G[v[i]]['neighbors'])/len(G[candidate]['neighbors'])) \
                v[i] = candidate if (np.random.rand() < len(G[candidate]['neighbors']))/len(G[v[i]]['neighbors']) \
                                 else v[i] 
            else:               # classic Random Walk
                v[i] = np.random.choice(G[v[i]]['neighbors'])
                
            if selected.get(v[i], 0) == 0:
                selected[v[i]] = 1
                size -= 1
                if size == 0:
                    flag = False
                    break   
            
    return selected.keys()
    
    
def most_important(G):
     """ returns a copy of G with
     the most important nodes
     according to the pagerank """ 
     ranking = nx.betweenness_centrality(G).items()
     print("Calculated ranks...")
     r = [x[1] for x in ranking]
     m = sum(r)/len(r) # mean centrality
     t = m*3 # threshold, we keep only the nodes with 3 times the mean
     Gt = G.copy()
     for k, v in ranking:
          if v < t:
              Gt.remove_node(k)
     return Gt

   
def disp(G, sd = 10, sz=512, m=True):
    s = time.time()
    G_nx = nx.Graph(create_adj_list(G))
    e = time.time()    
    print("Calculated adj_list in:" , (e-s))
    #G = nx.Graph(G)
    #G = most_important(G) 
    Gr = random_walk(G, seeds =sd, size = sz, metropolized = m )    
    print("Calculated important in:" , (time.time()-e))
    
    e = time.time()
    G_nx_sel = G_nx.subgraph(Gr)
    print("Calculated subgraph in:" , (time.time()-e))
    #app = Viewer(G)
    #app.mainloop()    
    
    pos = forceatlas2_layout(G_nx)
    #pos = nx.random_layout(G_nx)
    nx.draw_networkx_nodes(G_nx,pos,node_color='b',alpha=0.5,node_size=2)   
    nx.draw_networkx_edges(G_nx,pos,alpha=0.1)
    nx.draw_networkx_nodes(G_nx_sel,pos, node_color='r',alpha=0.5,node_size=20) 
    
    plt.plot(1)
    plt.axis('off')
    plt.show() 


## Now the layout function
def forceatlas2_layout(G, iterations=2, linlog=True, pos=None, nohubs=True,
                       kr=0.00001, k=None, dim=2, min_dist = 1):
    """
    Options values are
    g                The graph to layout
    iterations       Number of iterations to do
    linlog           Whether to use linear or log repulsion
    random_init      Start with a random position
                     If false, start with FR
    avoidoverlap     Whether to avoid overlap of points
    degreebased      Degree based repulsion
    """
    # We add attributes to store the current and previous convergence speed
    for n in G:
        G.node[n]['prevcs'] = 0
        G.node[n]['currcs'] = 0
        # To numpy matrix
    # This comes from the sparse FR layout in nx
    A = nx.to_scipy_sparse_matrix(G, dtype='f')
    nnodes, _ = A.shape

    try:
        A = A.tolil()
    except Exception as e:
        print(e)
        #A = (coo_matrix(A)).tolil()
    if pos is None:
        pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    if k is None:
        k = np.sqrt(1.0 / nnodes)
        # Iterations
    # the initial "temperature" is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = 0.3
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta ** 2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < min_dist, min_dist, distance)
            # the adjacency matrix row
            Ai = np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            Dist = k * k / distance ** 2
            if nohubs:
                Dist = Dist / float(Ai.sum(axis=1) + 1)
            if linlog:
                Dist = np.log(Dist + 1)
            displacement[:, i] += \
                (delta * (Dist - Ai * distance / k)).sum(axis=1)
        
        # update positions
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < min_dist, min_dist, length)
        pos += (displacement * t / length).T
        # cool temperature
        t -= dt
        # Return the layout
    return dict(zip(G, pos))


graph_set = []    
for d in dataset:
    ds = path+d
    print ("Reading dataset ", ds)
    graph_set, labels = load_data(ds)
    if d == 'proteins.graph':
        labels = labels[0]
    print ("Dataset: %s length: %s label distribution: %s"%(ds, len(graph_set), Counter(labels)))
    node_count = []
    for gidx, graph in graph_set.items():
        node_count.append(len(graph))
    print ("Avg #nodes: %s Median #nodes: %s Max #nodes: %s Min #nodes: %s"%(np.mean(node_count), np.median(node_count), max(node_count), min(node_count)))


for i in range( len(graph_set.keys())):
    if i%500 != 0:
        del graph_set[i]
    