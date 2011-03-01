# -*- coding: utf-8 -*-
from numpy import *

def center(x):
    ''' Remove the mean from the data. '''
    return x - mean(x, axis=0)

def normalize01(x, min=0, max=1):
    ''' Normalize the columns of a numpy array between 0 and 1. '''
    maxes = x.max(axis=0)
    mines = x.min(axis=0)
    return (x - mines[np.newaxis,:]) / (maxes - mines)[np.newaxis,:]

def standardize(x):
    ''' Zero mean unit variance'''
    #just subtract the mean if the standard deviation is zero or fail?
    return center(x) / std(x, 0)
