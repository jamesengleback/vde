import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import cat, relu, Tensor, LongTensor, zeros, linspace, from_numpy
from einops import rearrange
import prody as pr

from utils import mxn_sites_to_coords, mxn_ca_to_voxels, seq_to_dict, AAS
from data import Data


class ResLinear(nn.Module):
    def __init__(self, 
                 dim=32, 
                 ):
        super().__init__()
        self.c1 = nn.Linear(dim.sim)
    def forward(self, x):
        return x + relu(self.c1(x))

class ResConv3d(nn.Module):
    def __init__(self, 
                 dim=32, 
                 ):
        super().__init__()
        self.c1 = nn.Conv3d(in_channels=dim,
                            out_channels=dim,
                            kernel_size=(3,3,3),
                            padding=1,
                            stride=1)
        self.bn1 = nn.BatchNorm3d(dim)
    def forward(self, x):
        return self.bn1(x + relu(self.c1(x)))

class PredictionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(\
                            nn.Linear(dim,1),
                            nn.Dropout(0.2),
                            nn.ReLU())
    def forward(self,x):
        return self.head(x)


class Model(nn.Module):
    def __init__(self, 
                 ft_shape,
                 x_shape=None,
                 y_shape=None,
                 z_shape=None,
                 dim=32,
                 n_downsamples=6,
                 n_recycles=6,
                 n_pred_heads=3):
        super().__init__()
        self.hyper_params = {'ft_shape':ft_shape,
                             'x_shape':x_shape,
                             'y_shape':y_shape,
                             'z_shape':z_shape,
                             'dim':dim,
                             'n_recycles':n_recycles,
                             'n_downsamples':n_downsamples,
                             }
        self.c1 = nn.Conv3d(in_channels=ft_shape,
                            out_channels=dim,
                            kernel_size=(3,3,3),
                            stride=1)
        self.c2 = ResConv3d(dim)
        self.conv_downscale = nn.Sequential(\
                *[nn.Conv3d(in_channels=dim,
                            out_channels=dim,
                            kernel_size=(3,3,3),
                            stride=1),
                  nn.ReLU()])
        #downscale_size = 864
        downscale_size = self._get_downscaled_size()
        self.lin1 = nn.Sequential(nn.Linear(downscale_size, dim), nn.ReLU(), nn.Dropout(0.2))
        self.lin2 = nn.Sequential(nn.Linear(dim,dim),nn.ReLU(), nn.Dropout(0.2))
        self.pred_heads = nn.ModuleList([PredictionHead(dim) for i in range(n_pred_heads)])

    def forward(self, x):
        x_i = relu(self.c1(x))
        x_i = self.c2(x_i)
        for _ in range(self.hyper_params['n_downsamples']): # layer recycling
            x_i = self.conv_downscale(x_i)
        x_i = rearrange(x_i, 'b d x y z -> b (d x y z)')
        x_i = self.lin1(x_i) 
        for _ in range(self.hyper_params['n_recycles']):
            x_i = self.lin2(x_i)
        return [head(x_i) for head in self.pred_heads]

    def _get_downscaled_size(self):
        x_ = zeros(*[self.hyper_params[i] for i in ['ft_shape','x_shape','y_shape','z_shape']])
        x_ = rearrange(x_, 'd x y z -> () d x y z')
        x_i = relu(self.c1(x_))
        x_i = self.c2(x_i)
        for _ in range(self.hyper_params['n_downsamples']): # layer recycling
            x_i = self.conv_downscale(x_i)
        x_i = rearrange(x_i, 'b d x y z -> (b d x y z)')
        return x_i.shape[0]




def test():
    data = Data('../outputs/newscore2-98k.csv')
    #train_size =
    data_loader = DataLoader(data,
                            batch_size = 8,
                            shuffle = True)

    x, y = data[0]
    model = Model(*x.shape)

    from train import train
    train(model, data_loader, epochs=10, lr=1e-6)

if __name__ == '__main__':
    test()
