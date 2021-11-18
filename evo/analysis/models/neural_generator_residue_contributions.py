from math import prod
import pandas as pd
import torch
from torch import Tensor, LongTensor, cat, relu, sigmoid, zeros, randn, from_numpy
from torch.nn.functional import one_hot
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
from tqdm import tqdm

from simple_model import Data, AAS, Model
import sys
sys.path.append('../..')
from bm3 import BM3_DM

SAA = dict(zip(AAS.values(), AAS.keys()))

def ohe(seq):
    return one_hot(LongTensor([AAS[i] for i in seq]), 
                   num_classes=len(AAS)).float()

def eho(tensor):# shape : (b) l c
    if len(tensor.shape) == 2:
        ints = tensor.argmax(dim=-1)
        return ''.join([SAA[i.item()] for i in ints])
    else:
        return [eho(i) for i in tensor]


def ham_str(a, b):
    return sum([i != j for i,j in zip(a,b)])

def ham_tensor(a, b):
    #return (a.int() != b.int()).sum() / prod(a.shape)
    # shape : b l c -> b l
    a_, b_ = map(lambda tensor : tensor.argmax(dim=-1), (a, b))
    return (a_ == b_).sum(dim=-1)

class Generator(nn.Module):
    def __init__(self, 
                 channels,
                 l,
                 random_dim=16,
                 h=16,
                 *args):
        super().__init__()
        self.random_dim = random_dim
        self.layers = nn.Sequential(nn.Linear(random_dim,h),
                                    nn.Dropout(0.2),
                                    nn.ReLU(),
                                    nn.LayerNorm(h),
                                    #nn.Linear(h, h),
                                    #nn.Dropout(0.2),
                                    #nn.ReLU(),
                                    #nn.LayerNorm(h),
                                    nn.Linear(h, l*channels),
                                    nn.Dropout(0.2),
                                    nn.LayerNorm(l*channels),
                                    Rearrange('b (c l) -> b l c', l=l),
                                    nn.Softmax(dim=-1))
    def forward(self, batch_size=16):
        noise = randn(batch_size, self.random_dim)
        z = self.layers(noise)
        return z


def main(N_STEPS):
    BATCH_SIZE = 64
    data = Data('../outputs/all-data-new2.csv', test=True)
    DM_GENE = ''.join([BM3_DM[i] for i in data.mutation_sites])
    X = ohe(DM_GENE)

    model = torch.load('simple-model.pt')

    generator = Generator(l=len(data.mutation_sites), 
                          channels=len(AAS),
                          random_dim=2,
                          h=4)

    def ham_dist_to_dm(xh):
        return ham_tensor(repeat(X, 'l c -> b l c', b=xh.shape[0]), xh)

    opt = torch.optim.Adam(generator.parameters(), lr=1e-2)
    with tqdm(range(N_STEPS)) as n_steps:
        for i in n_steps:
            xh = generator(BATCH_SIZE)
            yh = cat([model(xh).unsqueeze(0) for _ in range(8)], dim=0)
            yh_aff, yh_dist = yh[:,:,0], yh[:,:,1]
            uncertainty_aff = yh_aff.std().mean()
            uncertainty_dist = yh_dist.std().mean()
            ham = ham_dist_to_dm(xh).float()
            ham.requires_grad_()
            loss = sum([ham.mean(),
                        yh_aff.mean(),
                        yh_dist.mean(),
                        uncertainty_aff.mean(),
                        uncertainty_dist.mean()]) / 6
            loss.backward()
            opt.step()
            opt.zero_grad()
            display = lambda t : round(t.detach().item(), 3)
            n_steps.set_postfix({'loss':display(loss.mean()),
                                 'ham':display(ham.mean()),
                                 'yh_aff':display(yh_aff.mean()),
                                 'yh_dist':display(yh_dist.mean()),
                                 'uncertainty_aff':display(uncertainty_aff.mean()),
                                 'uncertainty_dist':display(uncertainty_dist.mean()),
                                 })
    gen = eho(xh)
    df = pd.DataFrame([gen, 
                      [ham_str(i, DM_GENE) for i in gen],
                      #yh.mean(-1).detach().numpy()
                      ],
                      index = ['gene','ham'],#,'yh']
                      ).T
    print(df)




if __name__ == '__main__':
    main(int(sys.argv[1]))
