# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 01:19:38 2016

@author: yash

TODO:
[1] : add L2 regularisation
[2] : in g_pool flattening, make centrality independent of nx
[3]*: Pre allocate memory for adj matrices and adj_list
"""
from __future__ import print_function 
import numpy as np
import networkx as nx
from heapq import nlargest


class activation(object):
    #TODO make this super class for all classes implementing any type of layer
    def __init__(self, fn ='ReLu'):
        self.fn = fn
            
    def activate(self, inp):
        e = 2.718281828
        #clip the outputs, since size of receptive field can vary a lot
        ##RelU/Sigmoid/Softmax
        if self.fn == 'ReLu':
            return np.clip([max(0.01, item) for item in inp], -5, 5)
            
        elif self.fn == 'Sigmoid':
            return 1/(1+e**-inp)
            
        elif self.fn == 'Softmax':
            temp =  e**inp
            return temp/np.sum(temp)
            
        elif self.fn == 'Tanh':
            return (1-e**(-2*inp))/(1+e**(-2*inp))
            
        else:
            raise ValueError("Invalid activation function")
        
    
    def derivative(self, inp):
        if self.fn == 'ReLu':
            return np.array([1 if item>0.01 else 0.01 for item in inp])
            
        elif self.fn == 'Sigmoid':
            return inp*(1-inp)
            
        elif self.fn == 'Softmax':
            #hard coded for log likelihood loss
            #TODO: generic            
            return 1
            
        elif self.fn == 'Tanh':
            return 1-(inp**2)
            
        else:
            raise ValueError("Invalid activation function")
            
class g_cnn(object):
    
    def __init__(self,  prv_count = 1, stride = 1, filter_count = 4, size = 3, fn = 'ReLu'):
        #Layer variables
        self.stride         = stride  #not used, for time being
        self.filter_count   = filter_count
        self.size           = size
        self.filter         = np.random.randn(filter_count, size, prv_count)*np.sqrt(1/10000)  #TODO instead of 50 put fan-in     
        self.filter_del     = np.zeros(self.filter.shape)        
        self.bias           = np.random.randn(filter_count)*0.0001  
        self.bias_del       = np.zeros(self.bias.shape)
        self.temp           = np.zeros(size)
        self.act            = activation(fn)
        self.prv_count      = prv_count
        #Adam weight update variables
        self.m      = 0
        self.v      = 0
        self.beta1  = 0.9
        self.beta1t = 1
        self.beta2  = 0.999
        self.beta2t = 1
        self.alpha  = 0.00001
        self.eps    = 1e-8
    
    def forward(self, G):
        #print("forward gcnn!!")
        #TODO: **IMP** many dynamic arrays! pre-allocate them, use dict during run time, something!! 
        #nnodes is now fixed for each graph via sampling, no need for dynamc arrays!!
        #for each node, uses its 'val' and computes 'conv'
        nnodes = len(G.keys())
        
        #create dicts to map between actual and temporary node IDs
        self.idx_to_node = [node for node in G.keys()]#{i:j for i,j in enumerate(G.keys())}
        self.node_to_idx = {j:i for i,j in enumerate(G.keys())}
        
        #create adjacency matrix
        self.adj_mat = np.zeros((nnodes, nnodes))
        for i in G.keys():
            for j in G[i]['neighbors']:
                self.adj_mat[self.node_to_idx[i]][self.node_to_idx[j]] = 1
        
        #powers of adj matrix
        #to find no.of paths between i,j of length 'level'
        adj_mat_pow = np.array([np.identity(nnodes)]*self.size)        
        for level in range(1,self.size):
            adj_mat_pow[level] = (adj_mat_pow[level-1].dot(self.adj_mat))
        
        #keep path at level(i) only if level(j) == False, for all j < i
        adj_mat_pow = adj_mat_pow.astype(bool)
        for level in range(self.size -1, 0):
            temp = np.zeros((nnodes,nnodes)).astype(bool)
            for j in range(level-1, -1):
                temp += adj_mat_pow[j] 
                
            # x = x&~y, output =1, only when x==1, and y==0
            adj_mat_pow[level] &= np.invert(temp) 
        
        #create adj_list from the adj_matrix
        self.adj_list = [0]*self.size
        for level in range(self.size):
            self.adj_list[level] = [[j  for j in range(nnodes) if adj_mat_pow[level][i][j]] \
                                        for i in range(nnodes)]
                                            
        #for each node:
        #  if neighbors exists:
        #   for neighbors at each level of the node:
        #       find the values of neighbor
        #       these values are vectors of dim prv_size, corresponding to each prv filter
        #       sum over the values corresponding to same prv filter
        #  else: return [0]*prv_size                                
        #
        #   resultant is temp[level][prv_size]
        #   filter has 3 dim, [filter_no][level][prv_size]
        #
        #   multiply filter and temp
        #   sum over the axis of both: levels and prv_size
        #   activate the sum added with bias
        #   resulatant  = convolved value for that node for each filter
        for node in range(nnodes):
            self.temp = np.array([np.sum([G[self.idx_to_node[idx]]['val'] \
                                         for idx in self.adj_list[level][node]], axis = 0) \
                                 if len(self.adj_list[level][node]) !=0 \
                                 else np.zeros(self.prv_count) \
                        for level in range(self.size)] )
            #print(self.filter.shape, self.temp.shape, self.bias.shape, np.shape(G[self.idx_to_node[0]]['val']))
            G[self.idx_to_node[node]]['conv'] = self.act.activate \
                                                (np.sum(self.temp * self.filter, axis = (1,2))  + self.bias)
            #create dummy space to accumulate deltas later
            G[self.idx_to_node[node]]['delta'] = np.zeros(self.filter_count)
        return G    

    def backprop(self, G):
        #print("back gcnn")
        #for each node, use its 'delta' provided by the pooling layer and computes 'error'
        
        for node in G.keys():
            G[node]['delta'] = self.act.derivative(G[node]['conv'])*G[node]['delta']
               
        #backprop convolution similar to fwd convolution 
        #since filters are independent of orientation, no flipping is required
        #instead of 'val'  we use 'delta' 
        #instead of 'conv' we use 'error' to store the corresponding errors for the nodes
        nnodes = len(G.keys())
        for node in range(nnodes):
            self.temp = np.array([np.sum([G[self.idx_to_node[idx]]['delta'] \
                                         for idx in self.adj_list[level][node]], axis = 0) \
                                 if len( self.adj_list[level][node]) !=0 \
                                 else np.zeros(self.filter_count) \
                        for level in range(self.size)] )
            
            #transpose the filter to go back to dimension of 'prv_count'
            #calculate actual error for the nodes 
            #print(self.filter.T.shape, self.temp.shape)
            G[self.idx_to_node[node]]['error'] = np.sum(self.filter.T*self.temp , axis = (1,2))  
            
            #calculate deltas for filter
            self.filter_del += self.temp.reshape(self.filter_count, self.size,  1)*G[self.idx_to_node[node]]['val']
            
        self.bias_del = np.sum([G[node]['delta'] for node in G.keys()], axis = 0)
        
        return G
        
    def update(self, batch_size = 1):
        #Adam weight update
        self.filter_del /= batch_size
        self.bias_del   /= batch_size
        
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        self.m = self.beta1*self.m + (1 - self.beta1)*self.filter_del
        self.v = self.beta2*self.v + (1 - self.beta2)*(self.filter_del**2)
        
        rate = self.alpha*np.sqrt(1 - self.beta2t)/(1 - self.beta1t)
        self.filter -= rate*self.m/(np.sqrt(self.v) + self.eps)
        self.bias   -= self.alpha*self.bias_del  #(duh..) simple SGD update for bias :P
        
        #reset all deltas
        self.filter_del.fill(0)


class g_pool(object):
    
    def __init__(self, ratio = 2, flat = False):
        ##book-keeping variable for backprop error
        self.flat = flat
        self.ratio = ratio
        
    def upsample(self, G):
        """
        Upsample the error from the Graph at layer (l+1)
        to the graph at layer (l)        
        """
        #print("upsample ")
        if self.flat:
            G = self.unflatten(G)
            
        for node in G.keys():
            nb = self.G_old[node]['neighbors']
            err = G[node]['error']/(len(nb)+1)
            #print(self.G_old[nb[0]]['delta'].shape, err.shape, self.G_new[nb[0]['delta'].shape])
            for n in nb:
                self.G_old[n]['delta'] += err   #sum up fraction of errors as it may have been neighbor to more than one pooled node.
            self.G_old[node]['delta'] += err    #error contribution of the node itself
        return self.G_old
        
    def downsample(self, G):
        """
        For smplicity, trying to maintain a single graph structure for new Graph, 
        hence individual values from each filter can't be used to sample (like in images)
        since resultant of that can't be stacked up because
        of differnet structures that might arise from pooling each conv filter separately
        """
        #print("downsample")
        self.G_old = G
        ##[Method 1] Simplest way
        self.keep_count = len(G.keys())//self.ratio
        #TODO: try using max
        nodes = self.top_k(G, self.keep_count)
        self.val_len = len(G[nodes[0]]['conv'])
        G_new = {}
        for node in nodes:
            G_new[node] = {}
            #new neighbors = only prv neighbors who passed pooling step
            G_new[node]['neighbors'] = [n for n in G[node]['neighbors'] if n in nodes] #list(set(G[node]['neighbors']) & set(nodes))
            #val = mean of its neighbor's and itself's values            
            G_new[node]['val'] = np.mean([G[n]['conv'] for n in G[node]['neighbors']] + [G[node]['conv']], axis=0)
            
        if self.flat:
            return self.flatten(G_new)
            
        return G_new
    
    def top_k(self, G, k=200):
        #return top k nodes
        return nlargest(k, G.keys(), key = lambda e: np.mean(G[e]['conv']))
        
    def flatten(self, G):
        """
        make the graph flat based on centrality (anything better?) so that 
        fully connected nets can take it as an input
        
        centrality can be useful in tranferring values 
        in the apx labeling invariant way        
        """
        adj_list = {}
        for k, v in G.items():
            adj_list[k] = v['neighbors']
        
        #compute the centrality of each node
        central = nx.betweenness_centrality(nx.Graph(adj_list)).items()
        #compute node ordering based on centrality
        self.ranking = sorted(central, key = lambda item: item[1], reverse = True)
        
        #concatenate all the filter values of nodes in order of their ranking
        vec = [G[node[0]]['val'] for node in self.ranking]
        vec = np.reshape(vec, -1)
        if len(vec) == 504:
            print(len(G.keys()), self.ranking)
        return vec
        
    def unflatten(self, vec):
        """
        convert the linear error vector coming from 
        fully connected nets into graph structure
        """
        vec = np.reshape(vec, (-1, self.val_len))        
        G = {}
        for idx, node in enumerate(self.ranking):
            G[node[0]] = {'error' : vec[idx]}
            
        return G


class fc_nn(object):
    def __init__(self, nodes, prv, fn='ReLu', dropout = 1):
        """
        nodes   = Nodes in this layer
        prv     = Nodes in previous layers
        fn      = Output activation function
        dropout = Dropout probability ON THE INCOMING VECTOR
        """
        #TODO:add delta accumulator
        self.nodes    = nodes
        self.dropout  = dropout
        self.synapses = np.random.randn(nodes, prv)*(2.0/np.sqrt(prv)) 
        self.synapses_del = np.zeros(self.synapses.shape)
        self.bias     = np.random.random((nodes))*0.01 
        self.bias_del = np.zeros(self.bias.shape)
        self.act      = activation(fn)
        
        #Adam weight update variables
        self.m      = 0
        self.v      = 0
        self.beta1  = 0.9
        self.beta1t = 1
        self.beta2  = 0.999
        self.beta2t = 1
        self.alpha  = 0.0001
        self.eps    = 1e-8    
        
    def forward(self, inp):
        #TODO:assert input dimensions
        #add option for bayesian with dropouts for predicting confidence of results
        #print("forward FCNN")
        self.inp = inp
        
        if self.dropout != 1:
            mask = (np.random.rand(self.inp.shape) < self.dropout) / self.dropout 
            self.inp *= mask 
        #print(self.synapses.shape, self.inp.shape, self.bias.shape) 
        self.out = self.act.activate(self.synapses.dot(self.inp) + self.bias) 
        return self.out
    
    def backprop(self, error):
        ##
        #print("backprop FCNN")
        self.error = self.act.derivative(self.out)*error
        #print(self.error.shape, error.shape, self.inp.shape)
        self.synapses_del += self.error.reshape(self.nodes,1)*self.inp
        return self.synapses.T.dot(self.error)
        #return self.act.derivative(self.inp)*self.synapses.T.dot(error)
        
    def update(self, batch_size):
        #ADAM update
        self.synapses_del /= batch_size
        self.bias_del     /= batch_size
        
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        self.m = self.beta1*self.m + (1 - self.beta1)*self.synapses_del
        self.v = self.beta2*self.v + (1 - self.beta2)*(self.synapses_del**2)
        
        rate = self.alpha*np.sqrt(1 - self.beta2t)/(1 - self.beta1t)
        self.synapses   -= rate*self.m/(np.sqrt(self.v) + self.eps)
        self.bias       -= self.alpha*self.bias_del  #(duh..) simple SGD update for bias :P
        
        #reset all deltas after update
        self.synapses_del.fill(0)