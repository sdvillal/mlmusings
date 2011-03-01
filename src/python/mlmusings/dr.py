#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
from scipy.linalg import eig
from kernels import kernel_matrix, gaussian_kernel
import os
import prepro

def inertia(eigenvalues):
    return[e / sum(eigenvalues) for e in eigenvalues]

def select_top(eigenvalues, eigenvectors, to_retain=None):
    if not to_retain:
        to_retain = len(eigenvalues)
    indices = argsort(eigenvalues)[::-1][:to_retain]
    return real(eigenvalues[indices]), eigenvectors[:,indices]

def pca(x, target_dim=None):

    #Eigendecomposition of the covariance matrix of the centered data
    eigenvalues, eigenvectors = eig(cov(prepro.center(x).T))

    #Selection of the PCs
    eigenvalues, eigenvectors = select_top(eigenvalues, eigenvectors, target_dim)

    return dot(x, eigenvectors), eigenvectors, eigenvalues, inertia(eigenvalues)

def kernelmatrix(data,kernel,param=array([3,2])):

    if kernel=='linear':
        return dot(data,transpose(data))
    elif kernel=='gaussian':
        K = zeros((shape(data)[0],shape(data)[0]))
        for i in range(shape(data)[0]):
            for j in range(i+1,shape(data)[0]):
                K[i,j] = sum((data[i,:]-data[j,:])**2)
                K[j,i] = K[i,j]
        return exp(-K**2/(2*param[0]**2))
    elif kernel=='polynomial':
        return (dot(data,transpose(data))+param[0])**param[1]

def kernelpca(data,kernel,redDim):

    nData = shape(data)[0]
    nDim = shape(data)[1]

    K = kernelmatrix(data,kernel)

    # Compute the transformed data
    D = sum(K,axis=0)/nData
    E = sum(D)/nData
    J = ones((nData,1))*D
    K = K - J - transpose(J) + E*ones((nData,nData))

    # Perform the dimensionality reduction
    evals,evecs = eig(K)
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices[:redDim]]
    evals = evals[indices[:redDim]]

    sqrtE = zeros((len(evals),len(evals)))
    for i in range(len(evals)):
        sqrtE[i,i] = sqrt(evals[i])

    #print shape(sqrtE), shape(data)
    newData = transpose(dot(sqrtE,transpose(evecs)))

    return newData, K, evecs, evals, inertia(real(evals))

def kpca(x, target_dim=None, kernel=gaussian_kernel, **kernel_params):

    num_examples, num_features = shape(x)
    if not target_dim: #Recall that in kpca each point spans a direction in feature space and so there are numE components... we usually do not want that many
        target_dim = num_features

    #Kernel matrix
    K = kernel_matrix(x, kernel=kernel, **kernel_params)

    #Center the data in feature space: K' = K -1nK -K1n +1nK1n
    #Some algebra can optimize this both in time and space, think...
    unon = ones((num_examples, num_examples)) / num_examples
    unonK = dot(unon, K)
    K = K - unonK  - unonK.T + dot(unonK, unon)

    #Eigendecomposition of the centered kernel matrix data
    eigenvalues, eigenvectors = eig(K)

    #Selection of the PCs
    eigenvalues, eigenvectors = select_top(eigenvalues, eigenvectors, target_dim)

    #Transform
    x = dot(diag(sqrt(eigenvalues)), eigenvectors.T).T

    return x, K, eigenvectors, eigenvalues, inertia(eigenvalues)

#from pylab import *
#from numpy import *

def kpcasum(x, y, sigmas=[0.2, 0.6, 1.0, 1.4, 1.8, 2.2]):
    for sigma in sigmas:
        kpcax = kpca(x, kernel_param=sigma)
        acc = nn_acc(nns(kpcax, 3), y)
        vr = vizrank(kpcax, y)[0]
        print('sigma=%.1f,acc=%.4f,vr(%d,%d)=%.4f' % (sigma, acc, vr[1], vr[2], vr[0]))