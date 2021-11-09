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

def mxn_sites_to_coords(pdb_path='../../../data/4KEY.pdb'):
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

def mxn_ca_to_voxels(mxn_ca, seq, N_DIVISIONS=16):
    # mxn_ca : df - index: resnum ; cols [x,y,z]
    # seq : dict - {site:aa,...}
    #print(len(mxn_ca), len(seq))
    #assert len(seq) == len(mxn_ca) ### todo - find out why these don't match
    def min_max_scale(ca):
        mins = ca.min(axis=0)
        ca_ = ca.subtract(mins)
        maxs = ca_.max(axis=0)
        return ca_.div(maxs)
    mxn_ca_scaled = min_max_scale(mxn_ca)
    FEATURE_DIM = len(AAS)
    mxn_ca_scaled_ = mxn_ca_scaled * N_DIVISIONS
    #dim_lim = lambda x : round(x.max() + 5)
    grid = zeros(*[FEATURE_DIM] + [N_DIVISIONS + 1]*3 )
    for row,aa in zip(mxn_ca_scaled_.index, seq.values()):
        x,y,z = [round(i) for i in mxn_ca_scaled_.loc[row]]
        grid[AAS[aa],x,y,z] = 1. # one hot encoding
    return grid # shape : c x y z

def seq_to_dict(seq, sites):
    return dict(zip(sites, seq))

