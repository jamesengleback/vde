import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import cat, relu, Tensor, LongTensor, zeros, linspace, from_numpy
from einops import rearrange
import prody as pr

import sys
sys.path.append('../..')
import bm3
from bm3 import MXN_SITES

from utils import mxn_sites_to_coords, mxn_ca_to_voxels, seq_to_dict, AAS

class Data(Dataset):
    def __init__(self, 
                 path, 
                 template_path,
                 N_DIVISIONS=16,
                 test=False):
        super().__init__()
        self.path = path # csv
        self.template_path = template_path # pdb
        self.N_DIVISIONS = N_DIVISIONS
        if test:
            self.df = pd.read_csv(path, index_col=0, nrows=100).reset_index(drop=True)
        else:
            self.df = pd.read_csv(path, index_col=0).reset_index(drop=True)
        
        self.mxn_site_ca = mxn_sites_to_coords(self.template_path) # optional path to struc
        self.mxn_dicts, self.affinities, self.dists, self.scores = self.process_data(self.df)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        voxels = mxn_ca_to_voxels(self.mxn_site_ca, 
                                  self.mxn_dicts[idx],
                                  N_DIVISIONS=self.N_DIVISIONS)
        return voxels, self.affinities[idx],self.dists[idx], self.scores[idx]
    def process_data(self, df):
        def min_max_scale(t):
            t -= min(t.clone())
            t /= max(t.clone())
            return t
        def process_metric(df_col):
            t = from_numpy(df_col.values).float()
            t_scaled = min_max_scale(t)
            return rearrange(t_scaled, 'b -> b ()')

        affinities = process_metric(df['aff_mean'])
        dists = process_metric(df['dist_mean'])
        scores = process_metric(df['score'])
        # ---
        mxn_dicts = [seq_to_dict(i, MXN_SITES) for i in df.gene]
        return mxn_dicts, affinities, dists, scores
