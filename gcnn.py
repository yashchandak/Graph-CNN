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


def train(save_path = '', load_path = False):
    db_path = '/home/yash/Project/dataset/GraphSimilarity/reddit_multi_5K.graph'
    db = Data(db_path)
    class_count = db.classes
    inp_size    = db.val_size
    
    if load_path:
        net = load(load_path)
    else:
        net = [g_cnn(prv_count = inp_size, filter_count = 4), g_pool(), \
               g_cnn(prv_count = 4,        filter_count = 8), g_pool(flat=True), \
               fc_nn(prv = ((inp_size//2)//2)*8, nodes = 512), \
               fc_nn(prv = 512, nodes = 256, dropout = True), \
               fc_nn(prv = 256, nodes = class_count, fn="softmax") ]
          
    epoch = 100
    checkpoint = 5
    batch_size = 1
    train_error = np.zeros(epoch)
    valid_error = np.zeros(epoch//checkpoint)
    
    for i in range(epoch):
        while(db.has_more):
            
            data_batch = db.next_batch()
            for d in data_batch:
                e = train_step(net, d)
                train_error[i] += np.sum(np.abs(e))     
            
            train_error[i] /= batch_size
            update(net, batch_size)
            
        db.has_more = True
        if i % checkpoint == 0:
            data_batch = db.get_test()
            for d in data_batch:
                e = fwd_pass(net, db.get_test())
                valid_error[i//checkpoint] = np.sum(np.abs(e))
            valid_error[i//checkpoint] /= len(data_batch)    
            save(net, save_path)
            

train(save_path = '/home/yash/Project/Graph-CNN/logs/run1.net')







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
                
    