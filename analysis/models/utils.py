from math import sin, cos
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import cat, relu, Tensor, LongTensor, zeros, linspace, from_numpy
from einops import rearrange

import prody as pr

import sys
sys.path.append('../..')
import bm3
from bm3 import MXN_SITES

AAS = dict(zip('ACDEFGHIKLMNPQRSTVWY',range(100)))

def mxn_sites_to_coords(pdb_path='../../../data/4KEY.pdb', struc=None):
    if struc is None:
        struc = pr.parsePDB(pdb_path)
    coords = pd.DataFrame(struc.getCoords(), columns=[list('xyz')])
    feilds = ['name','resname','resnum','chain']
    for i in feilds:
        coords[i] = struc.getData(i)
    coords.columns = ['x', 'y', 'z'] + feilds
    coords = coords.loc[coords['chain'] == 'A',:]
    ca = coords.loc[coords['name'] == 'CA',:]
    mxn_ca = pd.concat([ca.loc[ca['resnum'] == i,:] for i in MXN_SITES])
    _mxn_ca = mxn_ca
    _mxn_ca.index = _mxn_ca['resnum']
    _mxn_ca = _mxn_ca[['x','y','z']]
    return _mxn_ca

def mxn_ca_to_voxels(mxn_ca, 
                     seq, 
                     encoding='ohe',
                     N_DIVISIONS=16):
    assert encoding in {'ohe','chem'}
    def min_max_scale(ca):
        mins = ca.min(axis=0)
        ca_ = ca.subtract(mins)
        maxs = ca_.max(axis=0)
        return ca_.div(maxs)
    mxn_ca_scaled = min_max_scale(mxn_ca)
    mxn_ca_scaled_ = mxn_ca_scaled * N_DIVISIONS
    if encoding == 'ohe':
        FEATURE_DIM = len(AAS)
        grid = zeros(*[FEATURE_DIM] + [N_DIVISIONS + 1]*3 )
        for row,aa in zip(mxn_ca_scaled_.index, seq.values()):
            #print(row, aa)
            x,y,z = [round(i) for i in mxn_ca_scaled_.loc[row]]
            grid[AAS[aa],x,y,z] = 1. # one hot encoding
    elif encoding == 'chem':
        pass
    return grid # shape : c x y z

def rotate(p, x, y, z):
    x_m = np.array([[1,0,0,0],
                    [0, cos(-x), -sin(x), 0],
                    [0, sin(-x), cos(-x),0],
                    [0,0,0,1]])
    y_m = np.array([[cos(-y), 0, sin(-y), 0],
                    [0,1,0,0],
                    [-sin(-y),0,cos(-y),0],
                    [0,0,0,1]])
    z_m = np.array([[cos(-z),-sin(-z),0,0],
                    [sin(-z), cos(-z),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
    for m in [x_m, y_m, z_m]:
        tx = pr.Transformation(m)
        p = tx.apply(p)
    return p

class Ohe:
    num_features = len(AAS)
    def encode(aa):
        t = zeros(len(AAS))
        t[aa] = 1.
        return t

def seq_to_dict(seq, sites):
    return dict(zip(sites, seq))

