# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:26:09 2016

@author: yash
"""

from __future__ import print_function
import time
import pickle
from collections import Counter
import numpy as np
import networkx as nx
from pylab import show
from networkx_viewer import Viewer

path = '/home/yash/Project/Graph-CNN/Data/'
#dataset = ['mutag.graph', 'ptc.graph', 'enzymes.graph', 'proteins.graph', 'nci1.graph', 'nci109.graph', 'collab.graph', 'imdb_action_romance.graph', 'reddit_iama_askreddit_atheism_trollx.graph', 'reddit_multi_5K.graph', 'reddit_subreddit_10K.graph']
dataset = ['reddit_multi_5K.graph']#'imdb_comedy_romance_scifi.graph']


        
class Data(object):
    def __init__(self, path):
        ##something
        self.has_more = True
        self.read_data(path)
        
    def read_data(path):
        
    def next_batch(self):
        ##something    

def load_data(ds_name):
    f = open(ds_name, "rb")
    data = pickle.load(f, encoding='latin1')
    graph_data = data["graph"]
    labels = data["labels"]
    labels  = np.array(labels, dtype = np.float)
    return graph_data, labels

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

def create_adj_list(graph):
    adj_list = {}
    for k, v in graph.items():
        adj_list[k] = v['neighbors']
        
    return adj_list
    
    
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

   
def disp(G):
    s = time.time()
    G = create_adj_list(G)
    e = time.time()
    print("Calculated adj_list in:" , (e-s))
    G = nx.Graph(G)
    G = most_important(G) 
    print("Calculated important in:" , (time.time()-e))
    
    app = Viewer(G)
    app.mainloop()    
    
    #pos = nx.spectral_layout(G)
    #nx.draw_networkx_nodes(G,pos,node_color='b',alpha=0.5,node_size=8)
    #nx.draw_networkx_edges(G,pos,alpha=0.1)
    #show()