#!/usr/bin/env python
import sys
import numpy as np
from numpy import concatenate as cat
import pandas as pd
from sklearn.manifold import TSNE

def ohe(gene, dim=0):
    AAS = dict(zip('ACDEFGHIKLMNPQRSTVWY', range(20)))
    fn = lambda string : [AAS[i] for i in string]
    o = []
    for i in gene:
        x = np.zeros((1,len(AAS)))
        x[0, fn(i)] = 1.
        o.append(x)
    return cat(o, axis=dim)

def main(args):
    for path in args:
        df = pd.read_csv(path)
        genes = df.gene
        x = cat(list(map(lambda gene : ohe(gene,1), df.gene)))
        tsne = TSNE()
        x_ = tsne.fit_transform(x)
        pd.DataFrame(x_).to_csv(sys.stdout)

if __name__ == '__main__':
    main(sys.argv[1:])
