from math import prod
import pandas as pd
from sklearn import preprocessing
import torch
from torch import Tensor, LongTensor, cat, relu, sigmoid, zeros, randn, from_numpy
from torch.nn.functional import one_hot
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
from tqdm import tqdm

AAS = dict(zip('ACDEFGHIKLMNPQRSTVWY',range(21)))

def std_scaler(x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x

def min_max_scaler(x):
    x -= x.min()
    x /= x.max()
    return x

class Data(Dataset):
    def __init__(self, 
                 path,
                 test=False,
                 ):
        super().__init__()
        self._process(path, test)  
    def _process(self, path, test=False):
        if test:
            self.df = pd.read_csv(path, delimiter='\t', nrows=1000)
        else:
            self.df = pd.read_csv(path, delimiter='\t').drop_duplicates()
        y1 = from_numpy(self.df['c20_dist_mean'].values).float()
        y2 = from_numpy(self.df['mean_affinity'].values).float()
        self.y1 = rearrange(min_max_scaler(std_scaler(y1)), 'n -> n ()')
        self.y2 = rearrange(min_max_scaler(std_scaler(y2)), 'n -> n ()')
        #---
        self.mutation_sites = [i for i,j in enumerate(zip(*self.df['sequence'])) if len(set(j)) > 1]
        self.seq = [''.join([i[j] for j in self.mutation_sites]) for i in self.df['sequence']]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.ohe(self.seq[idx]), self.y1[idx], self.y2[idx]

    def ohe(self, seq):
        return one_hot(LongTensor([AAS[i] for i in seq]), num_classes=len(AAS)).float()

class ResLin(nn.Module):
    def __init__(self, 
                 h, 
                 dropout=0.2,
                 activation=nn.ReLU):
        super().__init__()
        self.layer = nn.Linear(h, h)
        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()
    def forward(self, x):
        return self.norm(self.activation(self.dropout(self.layer(x))) + x)

class Model(nn.Module):
    def __init__(self,
                 shape,
                 h=32,
                 n_layers=3,
                 dropout=0.2,
                 ):
        super().__init__()
        self.layers = nn.Sequential(\
                nn.Linear(prod(shape), h),
                *[ResLin(h=h, dropout=dropout, activation=nn.LeakyReLU) for _ in range(n_layers)],
                nn.Linear(h,2),
                nn.Sigmoid())
    def forward(self, x):
        if len(x.shape) == 3:
            x_ = rearrange(x, 'b d l -> b (d l)')
        elif len(x.shape) == 2:
            x_ = rearrange(x, 'd l -> () (d l)')
        return self.layers(x_)

def plot(y,yh, title='out.png'):
    import matplotlib.pyplot as plt
    to_np = lambda tensor : tensor.detach().cpu().numpy() if type(tensor) is Tensor else tensor
    plt.style.use('dark_background')
    plt.figure(figsize=(10,10))
    plt.scatter(to_np(y), to_np(yh), s=0.6)
    plt.plot([0,1],[0,1])
    plt.xlabel('true')
    plt.ylabel('pred')
    plt.title(title)
    plt.savefig(title)
    plt.close()


def main():
    data = Data('../outputs/all-data-new2.csv', test=False)
    x0, y1_0, y2_0 = data[0]
    model = Model(x0.shape, h=64, n_layers=2, dropout=0.2)
    test_size = len(data) // 4
    train_size = len(data) - test_size
    train_data, test_data = random_split(data, (train_size, test_size))
    train_loader = DataLoader(train_data,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_data,
                             batch_size=128,
                             shuffle=True)
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_record = []
    for epoch in range(100):
        with tqdm(train_loader) as train_loader_:
            running_losses=[]
            for x_i, y1_i, y2_i in train_loader_:
                yh = model(x_i)
                y1_h, y2_h = yh[:,0].unsqueeze(-1), yh[:,1].unsqueeze(-1)
                loss = loss_fn(y1_h, y1_i) + loss_fn(y2_h, y2_i)
                loss.backward()
                opt.step()
                opt.zero_grad()
                loss_record.append(loss.detach().cpu().unsqueeze(0))
                
                running_losses.append(loss.unsqueeze(0))
                if len(running_losses) >= 100:
                    mean_loss = cat(running_losses).mean()
                    train_loader_.set_postfix({'loss':mean_loss.cpu().detach().item(),
                                               'epoch':epoch+1})
                    running_losses=[]

    loss_record = cat(loss_record)
    torch.save(loss_record, 'loss_record.pt')

    y1_is = []
    y1_hs = []
    y2_is = []
    y2_hs = []
    losses = []
    for x_i, y1_i, y2_i in tqdm(test_loader):
        with torch.no_grad():
            yh = model(x_i)
            y1_h, y2_h = yh[:,0].unsqueeze(-1), yh[:,1].unsqueeze(-1)
            loss = loss_fn(y1_h, y1_i) + loss_fn(y2_h, y2_i)
            y1_is.append(y1_i)
            y1_hs.append(y1_h)
            y2_is.append(y2_i)
            y2_hs.append(y2_h)

            losses.append(loss)

    y1_is = cat(y1_is)
    y1_hs = cat(y1_hs)
    y2_is = cat(y2_is)
    y2_hs = cat(y2_hs)
    losses = Tensor(losses)
    print(losses.mean())
    plot(y1_is, y1_hs, 'c20_preds-scatter.png')
    plot(y2_is, y2_hs, 'aff_preds-scatter.png')
    torch.save(model, 'simple-model.pt')
    torch.save(y1_is, 'y1_is.pt')
    torch.save(y1_hs, 'y1_hs.pt')
    torch.save(y2_is, 'y2_is.pt')
    torch.save(y2_hs, 'y2_hs.pt')
    

if __name__ == '__main__':
    main()
