# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 20:58:07 2016

@author: yash
"""
from __future__ import print_function, 
import numpy as np
import networkx as nx


class g_cnn(object):
    
    def __init__(self, name, inp_shape, out_shape, stride = 1, filter_count = 3, size = 3, prv_count = 1, fn = 'ReLu'):
        ##
        self.stride = stride  #not used, for time being
        self.filter_count = filter_count
        self.size = size
        self.filter = np.random.randn(filter_count, size, prv_count)*np.sqrt(2/50)  #TODO instead of 50 put fan-in     
        self.filter_del = np.zeros(self.filter.shape)        
        self.bias = np.random.randn(filter_count)  
        self.bias_del = np.zeros(self.bias.shape)
        self.temp = np.zeros(size)
        self.name = name
        self.adj = [0]*size
        self.act = activation(fn)
    
    def forward(self, G):
        #all nodes should have a value under the field 'val' 
        nnodes = len(G.keys())
        
        #create dicts to map between actual and temporary node IDs
        self.idx_to_node = {i:j for i,j in enumerate(G.keys())}
        self.node_to_idx = {j:i for i,j in enumerate(G.keys())}
        
        #create adjacency matrix
        self.adj_mat = np.zeros((nnodes, nnodes))
        for i in G.keys():
            for j in G[i]['neighbors']:
                self.adj_mat[node_to_idx[i]][node_to_idx[j]] = 1
        
        #k powers of adj matrix
        #to find no.of paths between i,j of length 'level'
        adj_mat_pow = np.array([np.identity(nnodes)])        
        for level in range(1,self.size):
            adj_mat_pow.append(adj_mat_pow[level-1].dot(self.adj_mat))
        
        #keep path at level K(p) only if K(q) == False, for all q < p
        adj_mat_pow = adj_mat_pow.astype(bool)
        for level in range(self.size -1, 0):
            temp = np.zeros((nnodes,nnodes)).astype(bool)
            for j in range(level-1, -1):
                temp += adj_mat_pow[j] 
                
            # x = x&~y, output =1, only when x==1, and y==0
            adj_mat_pow[level] &= np.invert(temp) 
        
        #create adj_list from the adj_matrix
        self.adj_list = np.zeros(self.size)
        for level in range(self.size):
            self.adj_list[level] = [[j  for j in range(nnodes) if adj_mat_pow[level][i][j]] \
                                        for i in range(nnodes)]
                                            
        #for each node:
        #   for neighbors at each level of the node:
        #       find the values of neighbor
        #       these values are vectors of dim prv_size, corresponding to each prv filter
        #       sum over the values corresponding to same prv filter
        #
        #   resultant is temp[level][prv_size]
        #   filter has 3 dim, [filter_no][level][prv_size]
        #
        #   multiply filter and temp
        #   sum over the axis of both levels and prv_size
        #   activate the sum added with bias
        #   resulatant  = convolved value for that node for each filter
        for node in range(nnodes):
            self.temp = [np.sum([G[self.idx_to_node[idx]]['val'] \
                                 for idx in self.adj_list[level][node]], \
                        axis = 0) \
                        for level in range(self.size)] 
            
            G[self.idx_to_node[node]]['conv'] = self.act.activate \
                                                (np.sum(self.temp * self.filter, axis(1,2)) + self.bias)
        
            
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
                
    
    def backprop(self, G):
        ##all nodes of G should have 'delta'

        #calculate derivatives of activation for each node
        for key in G.keys():
            G[key]['delta'] = self.act.derivative(G[key]['delta'])
            
        #calculate actual error for the nodes        
        #backprop convolution similar to fwd convolution 
        #since filters are independent of orientation, no flipping is required
        #instead of 'val'  we use 'delta' given by pooling layer
        #instead of 'conv' we use 'error' to store the corresponding errors for the nodes
        for node in range(nnodes):
            self.temp = [np.sum([G[self.idx_to_node[idx]]['delta'] \
                                 for idx in self.adj_list[level][node]], \
                        axis = 0) \
                        for level in range(self.size)] 
            
            #transpose the filter to go back to dimension of 'prv_count'
            G[self.idx_to_node[node]]['error'] = np.sum(self.temp * self.filter.T, axis(1,2))  
            
                
        ##calculate deltas for filter
        
        
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
    #TODO make this super class for all classes implementing any type of layer
    def __init__(self, fn ='ReLu'):
        self.fn = fn
        
    def activate(self, inp):
        ##RelU/Sigmoid/Softmax
    
    def derivative(self, inp):
        ##ReLu/Sigmoid/Softmax


class Data(object):
    def __init__(self, path):
        ##something
        self.has_more = True
        self.read_data(path)
        
    def read_data(path):
        
    def next_batch(self):
        ##something


class fc_nn(object):
    def __init__(self, nodes, fn='ReLu', dropout = False):
        ##
    
    def forward(self):
        ##check if input is flat, otherwise reshape
    
    
    def backprop(self):
        ##
    
    def update(self):
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


def train(load_path = False):
    if load_path:
        net = load(load_path)
    else:
        net = [g_cnn(name = 'G1', filter_count = 4), g_pool(), \
               g_cnn(name = 'G2', filter_count = 8, prv_count = 4), g_pool(), \
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