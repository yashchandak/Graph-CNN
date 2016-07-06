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

        
class Data(object):
    has_more = True
    counter = 0
    val_size = 1 #size of the value assigned to each node initially
    def __init__(self, path, ratio = 0.9, batch_size = 1):
        self.batch_size = batch_size
        self.ratio = ratio
        
        self.read_data(path)
        
    def read_data(self, path):
        print ("Reading dataset ", path)
        self.graph_set, self.labels = self.load_data(path)
        #remove this
        for i in range(10,len(self.graph_set.keys())): del self.graph_set[i]
            
        if path.split('/')[-1] == 'proteins.graph': #exception case
            self.labels = self.labels[0]
        
        node_count = []
        for gidx, graph in self.graph_set.items():
            node_count.append(len(graph))
        
        #Data stats
        self.min, self.max     = min(node_count), max(node_count)
        self.mean, self.median = np.mean(node_count), np.median(node_count)
        self.len, self.classes = len(self.graph_set), len(Counter(self.labels)) 
        
        self.size = pow(2, int(np.log2(self.median))) #no.of vertices to keep from each graph
        self.shuffled          = np.arange(int(self.len*self.ratio))
        np.random.shuffle(self.shuffled)
        
        print ("Dataset: %s length: %s label distribution: %s"%(path, self.len, Counter(self.labels)))
        print ("Avg #nodes: %s Median #nodes: %s Max #nodes: %s Min #nodes: %s"%(self.mean, self.median, self.max, self.min))
        print ("Sampling size: ", self.size)
        
    def load_data(self, path):
        f = open(path, "rb")
        data = pickle.load(f, encoding='latin1')
        graph_data = data["graph"]
        labels = data["labels"]
        labels  = np.array(labels, dtype = np.int)
        return graph_data, labels

    def next_batch(self):
        data = []
        for idx in range(self.batch_size):
            c = self.shuffled[self.counter]                     #get next value in the shuffled list
            g = self.sampling(self.graph_set[c], self.size)   #sample the graph to a fixed vertex size graph
            
            #one-hot encoding
            truth = np.zeros(self.classes)
            truth[self.labels[c]] = 1
            
            self.counter += 1
            if self.counter == int(self.ratio*self.len): 
                self.counter = 0
                self.has_more = False
                print("Finished iterating over entire dataset, starting again...")
                
            data.append((g,truth))
        return data
                
    def get_test(self):
        data = []
        for idx in range(int(self.ratio*self.len), self.len):
            g = self.sampling(self.graph_set[idx], self.size) 
            truth = np.zeros(self.classes)
            truth[self.labels[c]] = 1       #one-hot encoding
            data.append((g,truth))
        return data
    
    def sampling(self, G, size): 
        """
        Returns a sampled graph from 'G' with no. of vertices = 'size'       
        """
        node_count = len(G.keys())
            
        if size > node_count:
            #add dummy nodes if graph has less nodes than required
            for i in range(node_count, size):
                G[i] = {'neighbors':[]}
        
        elif size <  node_count:    
            #sample using random walk if graph has more nodes than required
            selected = self.random_walk(G, size)
            G = self.subgraph(G, selected)
            
        #assign initial values to each node
        # a) default, 1
        # b) based on label of vertices?
        # c) normalised degree?
        #change self.val_size accodingly
        for v in G.keys():
            G[v]['val'] = [1] #default
            
        return G
    
    
    def subgraph(self, G, selected):
        """
        Generate an induced sub graph of G  
        using all nodes from list of 'selected' vertices
        """
        G_new = {}
        for node in selected:
            neighbors = [v for v in G[node]['neighbors'] if selected.get(v, 0)!=0]
            G_new[node] = {'neighbors':neighbors}        
        return G_new
    
    
    def random_walk(self, G, size, seeds=20, start_node=None,  metropolized=True):
        #TODO: disconnectednss can be a problem here, visualise the generate graph to fine tune algo
        if start_node==None:
            start_node = np.random.choice(list(G.keys()), seeds)
            
        v = start_node
        flag = True
        selected = {} #book-keeping of selected vertices
        
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
                
        return selected
        
    def create_adj_list(self,graph):
        print("Started creating adj_list...")
        adj_list = {}
        for k, v in graph.items():
            adj_list[k] = v['neighbors']
            
        print("Done")
        return adj_list


#######################################
#for random debugging and visualisation    
    
def most_important(G):
     """ 
     returns a copy of G with
     the most important nodes
     according to the pagerank
     """ 
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

   
def disp(G, obj, **args):
    s = time.time()
    G_nx = nx.Graph(obj.create_adj_list(G))
    e = time.time()    
    print("Calculated adj_list in:" , (e-s))
    #G = nx.Graph(G)
    #G = most_important(G) 
    Gr = obj.random_walk(G, args)    
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

db_path = '/home/yash/Project/dataset/GraphSimilarity/reddit_multi_5K.graph'
db = Data(db_path)

graphs = [] 
d = None
def main():
    global graphs, d
    d = Data(path+'/'+dataset[0])    
    for i in range( len(d.graph_set.keys())):
        if i%500 != 0:
            del d.graph_set[i]
    graphs = d.graph_set
    
#if '__name__' == '__main__':
#    main()
        