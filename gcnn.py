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

from layers import g_cnn, fc_nn, g_pool
#from data import Data
import data
from net import fwd_pass, train_step, update, save, load, calc_error

db = data.db

def train(save_path = '', load_path = False):
#    db_path = '/home/yash/Project/dataset/GraphSimilarity/reddit_multi_5K.graph'
#    db = Data(db_path)
    class_count = db.classes
    val_size    = db.val_size
    inp_size    = db.size
    
    if load_path:
        net = load(load_path)
    else:
        net = [g_cnn(prv_count = val_size, filter_count = 4), g_pool(), \
               g_cnn(prv_count = 4,        filter_count = 8), g_pool(flat=True), \
               fc_nn(prv = ((inp_size//2)//2)*8, nodes = 1024), \
               fc_nn(prv = 1024, nodes = 256, dropout = True), \
               fc_nn(prv = 256, nodes = class_count, fn="Softmax") ]
          
    epoch = 25
    checkpoint = 10
    batch_size = 1
    train_error = np.zeros(epoch*5000//batch_size, np.float)
    valid_error = np.zeros(epoch)
    ctr = 0
    for i in range(epoch):
        while(db.has_more):
            
            data_batch = db.next_batch()
            for d in data_batch:
                e = train_step(net, d)
                #print(e)
                train_error[ctr] += np.sum(np.abs(e))     
            
            #train_error[ctr] /= batch_size
            update(net, batch_size)
            
            ctr += 1
            if ctr%checkpoint == 0:
                print("Batch [%d]: Training Error: [%f]" %(ctr, train_error[ctr]))
        
        db.has_more = True
        data_batch = db.get_test()
        for d in data_batch:
            pred = fwd_pass(net, d[0])
            valid_error[i] += -np.sum(calc_error(pred, d[1]))
        valid_error[i] /= len(data_batch)    
        
        save(net, save_path)
        print("Epoch [%d]> Validation Error: [%f]" %(i, valid_error[i]))
            

train(save_path = '/home/yash/Project/Graph-CNN/logs/run1.net') 