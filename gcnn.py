# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 20:58:07 2016

@author: yash

TODO
[1] : Try keeping only single object of graph and appending additional values at each layer
[2] : Visualise the pooling result at each layer, I'm afraid it might get all disjoint or clique
[3] : add batch input support
"""
from __future__ import print_function 
import numpy as np
import networkx as nx
from heapq import nlargest

from layers import g_cnn, fc_nn, g_pool
from data import Data
from net import fwd_pass, train_step, update, save, load


def sampling():
    print("Sample representative graph form big graph")

def train(save_path = '', load_path = False):
    class_count = 10
    
    if load_path:
        net = load(load_path)
    else:
        net = [g_cnn(filter_count = 4), g_pool(), \
               g_cnn(filter_count = 8, prv_count = 4), g_pool(flat=True), \
               fc_nn(nodes = 256), \
               fc_nn(nodes = 512, dropout = True), \
               fc_nn(nodes = class_count, fn="softmax") ]
          
    epoch = 1500
    checkpoint = 25
    batch_size = 1
    train_error = np.zeros(epoch)
    valid_error = np.zeros(epoch//checkpoint)
    path = ""
    data = Data(path = path, batch_size = batch_size )
    
    for i in range(epoch):
        while(data.has_more):
            
            db = data.next_batch()
            for d in db:
                train_error[i] += train_step(net, d)                
            update(net, batch_size)
            
        if i % checkpoint == 0:
            valid_error[i//checkpoint] = fwd_pass(net, data.training)
            save(net, save_path)
            

train()







#       METHOD WITHOUT ADJ MATRIX COMPUTATON
#       TODO: DOESNT TAKE INTO ACCOUNT ALL PRV CONVOLUTION LAYER
#        for node in G.keys():
#            self.temp.fill(0)
#            adj[i] = [node]
#            for i in range(self.size):
#                if i : adj = list(set([n  for item in adj   for n in G[item]['neighbors'] ])    #neighbors at level i from the node
#                self.temp[i] += sum([G[item]['val'] for item in adj])                           #sum of neighbor's values
#                
#            G[node][self.name] = np.sum(self.temp * self.filter, axis = 1)
                
    