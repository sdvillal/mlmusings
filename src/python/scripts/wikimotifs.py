#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys #TODO: check PEP 366
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mlmusings
from mlmusings import neighbors as nn
from mlmusings import dr

PETECAN_ROOT = os.path.join(os.path.expanduser('~'), 'Proyectos', 'data', 'wikipedia-motifs')
ORIGINAL_ARFF = os.path.join(PETECAN_ROOT, 'ArticleEgoMotifCounts.arff')

def kpcasum(x, y, sigmas=[0.2, 0.6, 1.0, 1.4, 1.8, 2.2]):
    for sigma in sigmas:
        kpcax = dr.kpca(x, sigma=sigma)[0]
        acc = nn.nn_acc(nn.nns(kpcax, 3), y)
        vr = nn.vizrank(kpcax, y)[0]
        print('sigma=%.1f,acc=%.4f,vr(%d,%d)=%.4f' % (sigma, acc, vr[1], vr[2], vr[0]))

def main(datafile):
    _, _, _, x, y = mlmusings.io.load_arff(datafile)
    print('3-nearest-neighbor accuracy on original data: %.4f' % nn.nn_acc(nn.nns(x, 3), y))
    print('KernelPCA...')
    kpcasum(x, y)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ORIGINAL_ARFF)