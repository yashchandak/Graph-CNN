# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 01:43:16 2016

@author: yash
"""
import pickle
from layers import g_cnn, g_pool, fc_nn
import numpy as np

def fwd_pass(net, inp):
    #do one complete fwd pass
    for layer in net:
        if isinstance(layer, g_pool):
            inp = layer.downsample(inp)
        elif isinstance(layer, [g_cnn, fc_nn]):
            inp = layer.forward(inp)
        else:
            raise ValueError("Invalid layer type...")
    
    return inp #return the final output


def backprop(net, err):
    #one complete bwd pass
    for layer in reversed(net):
        if isinstance(layer, g_pool):
            err = layer.upsample(err)
        elif isinstance(layer, [g_cnn, fc_nn]):
            err = layer.backprop(err)
    
    return err #returns err at the initial input state, which is useless

def update(net, batch_size):
    #update the net parameters
    for layer in net:
        if not isinstance(layer, g_pool): layer.update()
    

def cross_entropy(pred, truth):
    print("TODO")    
    
def MSE(pred, truth):
    print("TODO")
    
def train_step(net, data, fn='cross_entropy'):
    inp, truth = data[0], data[1]
    out = fwd_pass(net, inp)
    
    if fn == 'cross_entropy':
        error = cross_entropy(out, truth)
    elif fn == 'MSE':
        error = MSE(out, truth)
    else:
        raise ValueError("Invalid loss function...")
        
    backprop(net, error) 
    return np.sum(np.abs(error))

def save(net, name):
    pickle.dump(net, open(name, "wb"))

def load(name):
    return pickle.load(open(name, "rb"))