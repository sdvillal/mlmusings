# -*- coding: utf-8 -*-
from numpy import *
from scipy.linalg import eig
from kernels import kernel_matrix, gaussian_kernel
import os
import prepro

def inertia(eigenvalues):
    return[eigenvalue / sum(eigenvalues) for eigenvalue in eigenvalues]

def select_top(eigenvalues, eigenvectors, to_retain=None):
    if not to_retain:
        to_retain = len(eigenvalues)
    indices = argsort(eigenvalues)[::-1]
    inertias = inertia(real(eigenvalues[indices]))
    indices = indices[:to_retain]
    return real(eigenvalues[indices]), real(eigenvectors[:,indices]), inertias

def pca(x, target_dim=None):

    #Eigendecomposition of the covariance matrix of the centered data
    eigenvalues, eigenvectors = eig(cov(prepro.center(x).T))

    #Selection of the PCs
    eigenvalues, eigenvectors, inertias = select_top(eigenvalues, eigenvectors, target_dim)

    return dot(x, eigenvectors), eigenvectors, eigenvalues, inertias

def kpca(x, target_dim=None, kernel=gaussian_kernel, **kernel_params):

    #Recall that in kpca, depending on the kernel, each point can span a new direction in feature space...
    num_examples, num_features = shape(x)
    if not target_dim:
        target_dim = num_features #...we usually do not want that many

    #Kernel matrix
    K = kernel_matrix(x, kernel=kernel, **kernel_params)

    #Center the data in feature space: K' = K -1nK -K1n +1nK1n
    #Some algebra can optimize this both in time and space, think...
    onen = ones((num_examples, num_examples)) / num_examples
    onenK = dot(onen, K)
    K = K - onenK  - onenK.T + dot(onenK, onen)

    #Eigendecomposition
    eigenvalues, eigenvectors = eig(K)

    #Selection of the PCs
    eigenvalues, eigenvectors, inertias = select_top(eigenvalues, eigenvectors, target_dim)

    #Transform
    x = dot(diag(sqrt(eigenvalues)), eigenvectors.T).T

    return x, K, eigenvectors, eigenvalues, inertias