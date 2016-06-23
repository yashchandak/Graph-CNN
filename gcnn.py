# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 20:58:07 2016

@author: yash
"""

import numpy as np
import networkx as nx


class g_cnn(object):
    
    def __init__(self, name, inp_shape, out_shape, stride = 1, filter_count = 3, size = 3, fn = 'ReLu'):
        ##
        self.stride = stride  #useless for time being
        self.filter_count = filter_count
        self.size = size
        self.filter = np.random.randn(filter_count, size)*np.sqrt(2/50)  #TODO instead of 50 put fan-in     
        self.bias = np.random.randn(filter_count)        
        self.temp = np.zeros(size)
        self.name = name
        self.adj = [0]*size
    
        self.init_weights()
    def init_weights(self):
        ##
    
    def forward(self, G):
        nnodes = len(G.keys())
        
        self.idx_to_node = {i:j for i,j in enumerate(G.keys())}
        self.node_to_idx = {j:i for i,j in enumerate(G.keys())}
        
        self.adj_mat = np.zeros((nnodes, nnodes)
        for i in G.keys():
            for j in G[i]['neighbors']:
                self.adj_mat[node_to+idx[i]][node_to_idx[j]] = 1
        
        #k powers of adj matrix
        #to find no.of paths between i,j of length 'level'
        self.adj_mat_pow = np.array([np.identity(nnodes)])        
        for level in range(1,size):
            self.adj_mat_pow.append(self.adj_mat_pow[level-1].dot(self.adj_mat))
        
                #keep path at level K(p) only if K(q) == False, for all q < p
        self.adj_mat_pow = self.adj_mat_pow.astype(bool)
        for level in range(size -1, 0):
            temp = np.zeros((nnodes,nnodes)).astype(bool)
            for j in range(level-1, -1):
                temp += self.adj_mat_pow[j]                
            self.adj_mat_pow[level] &= np.invert(temp) # x = x&~y, output =1, only when x==1, and y==0
        
        self.adj_list = np.zeros(size)
        for level in range(size):
            self.adj_list[level] = [[j  for j in range(nnodes) \
                                        if adj_mat_pow[level][i][j]]\
                                        for i in range(nnodes)]
                                            
                    
        
        
            
        
        for node in G.keys():
            self.temp.fill(0)
            adj[i] = [node]
            for i in range(self.size):
                #TODO precompute neighbours at different levels
                if i : adj = list(set([n  for item in adj   for n in G[item]['neighbors'] ])    #neighbors at level i from the node
                self.temp[i] += sum([G[item]['val'] for item in adj])                           #sum of neighbor's values
                
            G[node][self.name] = np.sum(self.temp * self.filter, axis = 1)
                
    
    def backprop(self, err):
        ##
        
    def update(self):
        ##add deltas to the weights


class g_pool(object):
    
    def __init__(self, switch = 1):
        ##book-keeping variable for backprop error
    
    def upsample(self):
        ##
    
    def downsample(self):
        ##create graph with new reduced nodes and edges
    
    def top_k(G, k=200):
        #return top 200 nodes
        


class activation(object):
    
    def __init__(self, fn ='ReLu'):
        self.fn = fn
        
    def activate(self, inp, ):
        ##RelU and Sigmoid
    
    def derivative(self, inp):
        ##ReLu and Sigmoid


class Data(object):
    def __init__(self):
        ##something
        
    def next_batch(self):
        ##something


class fc_nn(object):
    def __init__(self, nodes, fn='ReLu', dropout = False):
        ##
    
    def forward(self):
        ##check if input is flat, otherwise reshape
    
    
    def backprop(self):
        ##


def fwd_pass(net):
    #do one complete fwd pass

def backprop(net):
    #one complete bwd pass

def train_step(net, data):
    #fwd and bwd pass    
    
def update(net):
    #update the net parameters

def save():
    ##

def load():
    ##


def train():    
    net = [g_cnn(name = 'G1'), g_pool(), \
           g_cnn(name = 'G2'), g_pool(), \
           fc_nn(nodes = 256), \
           fc_nn(nodes =512, dropout = True), \
           fc_nn(nodes = class_count, fn="softmax") ]
          
    epoch = 1500
    checkpoint = 25
    train_error = np.zeros(epoch)
    valid_error = np.zeros(epoch//checkpoint)
    path = ""
    data = Data(path = path)
    
    for i in range(epoch):
        while(data.has_more):
            
            db = data.next_batch()
            for d in db:
                train_step(net, d)                
            update(net)
            
        if i % checkpoint == 0:
            valid_error = fwd_pass(net, data.training)
            save()
            

train()