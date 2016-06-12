# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 20:58:07 2016

@author: yash
"""

import numpy as np


class g_cnn(object):
    
    def __init__(self, inp_shape, out_shape, stride = 1, filter_count = 3, size = 3):
        ##
        self.stride = stride
        self.filter_count = filter_count
        self.size = size
        self.init_weights()
    
    def init_weights(self):
        ##
    
    def forward(self):
        ##
    
    def backprop(self):
        ##
        


class g_pool(object):
    def __init__(self):
        ##book-keeping variable for backprop error
    
    def upsample(self):
        ##
    
    def downsample(self):
        ##create graph with new reduced nodes and edges
        


class activation(object):
    
    def __init__(self, fn ='ReLu'):
        self.fn = fn
        
    def activate(self, inp, ):
        ##RelU and Sigmoid
    
    def derivative(self, inp):
        ##ReLu and Sigmoid