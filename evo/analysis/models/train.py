import sys
import os
import os.path as osp
import random
from string import ascii_lowercase
import json
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import cat, relu, Tensor, LongTensor, zeros, linspace, from_numpy
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from utils import mxn_sites_to_coords, mxn_ca_to_voxels, seq_to_dict, AAS
from data import Data
from model import Model

import argparse

def random_rotate(tensor): # todo
    return tensor

def train(model, 
          data_loader, 
          out_path,
          test_loader=None, # for scoring on checkpoints
          epochs=10,
          learning_rate=1e-6,
          tensorboard=False,
          checkpoint=False,
          outdir=None,
          cuda=False,
          quiet=False,
          ):
    # defaults
    # ---
    if checkpoint:
        os.makedirs(osp.join(out_path, 'checkpoints'), exist_ok=True)
    if tensorboard:
            writer = SummaryWriter(osp.join(out_path,'tensorboard'), flush_secs=1)
    if cuda:
        model.cuda()

    # train loop
    # --- 
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        with tqdm(data_loader) as data_loader_:
            for i, batch in enumerate(data_loader_):
                if cuda:
                    map(lambda t : t.cuda(), batch)                
                # ---
                v, a, d, s = batch # voxels, affinity, distance, score
                v_ = random_rotate(v)
                y_h = model(v_)
                a_h, d_h, s_h = y_h[0], y_h[1], y_h[2]
                loss_a = loss_fn(a, a_h)
                loss_d = loss_fn(d, d_h)
                loss_s = loss_fn(s, s_h)
                # ---
                loss = sum([loss_a, loss_d, loss_s])
                loss.backward()
                opt.step()
                opt.zero_grad()
                data_loader_.set_postfix({\
                                'loss_a':loss_a.detach().item(),
                                'loss_d':loss_d.detach().item(),
                                'loss_s':loss_s.detach().item(),
                                'loss'  :loss.detach().item(),
                                })
                # ---
                if tensorboard:
                    global_step = i + (len(data_loader) * epoch)
                    writer.add_scalars(main_tag='run',
                            tag_scalar_dict={"loss_a":loss_a,
                                             "loss_d":loss_d,
                                             "loss_s":loss_s,
                                             "loss":loss},
                            global_step=global_step)

                    writer.flush()
                # ---
        if checkpoint:
            if test_loader is not None:
                metrics = test(model, 
                               test_loader,
                               plot=True,
                               plot_path=osp.join(out_path, f'model_epoch_{epoch}.png')
                               )
                with open(osp.join(out_path, 'checkpoints', 'checkpoints.txt'), 'a') as f:
                    f.write(f'model_epoch_{epoch}.pt : {metrics}')
            torch.save(model, osp.join(out_path, f'model_epoch_{epoch}.pt'))

def test(model, 
        data_loader,
        cuda=False,
        plot=False,
        plot_path=None
        ):
    aff, dist, score = [], [], []
    aff_h, dist_h, score_h = [], [], []
    if cuda:
        model = model.cuda()
    with torch.no_grad():
        for v_i, a_i, d_i, s_i in tqdm(data_loader):
            if cuda:
                map(lambda t : t.cuda(), [v_i, a_i, d_i, s_i])
            yh = model(v_i)
            a_h, d_h, s_h = yh[0], yh[1], yh[2]

            aff.append(a_i)
            dist.append(a_i)
            score.append(a_i)

            aff_h.append(a_h)
            dist_h.append(d_h)
            score_h.append(s_h)

    aff, dist, score, aff_h, dist_h, score_h = map(lambda l : cat(l), [aff, dist, score, aff_h, dist_h, score_h])
    loss_fn = nn.MSELoss()
    metrics = {
            'loss_aff':loss_fn(aff,aff_h).detach().cpu().item(),
            'loss_dist':loss_fn(dist,dist_h).detach().cpu().item(),
            'loss_score':loss_fn(score,score_h).detach().cpu().item(),
            }
    if plot:
        plot_scatter(*zip(\
                ['aff', 'dist', 'score'],
                [aff, dist, score],
                [aff_h, dist_h, score_h]),
                plot_path=plot_path)
    return metrics

def plot_scatter(*args, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(args), 1, figsize=(5,10))
    for ax_i, (name, x_i, y_i) in zip(ax, args):
        ax_i.scatter(x_i, y_i)
        ax_i.plot([0,1],[0,1])
        ax_i.set_title(name)
        ax_i.set_xlabel('Actual')
        ax_i.set_ylabel('Predicted')
        ax_i.set_ylim(-0.1,1)
        ax_i.set_xlim(-0.1,1)
    plt.tight_layout()
    if 'plot_path' in kwargs: # bad argument handling
        if kwargs['plot_path'] is not None:
            plt.savefig(kwargs['plot_path'])
        else:
            plt.show()
    else:
        plt.show()

def write_config(args, out_path):
    with open(osp.join(out_path, 'config.json'),'w') as f:
        d = args.__dict__
        config = {str(i):str(j) for i, j in zip(d.keys(), d.values())}
        json.dump(config, f)


def main(args):
    if args.outdir is None:
        outdir = ''.join(random.choices(ascii_lowercase, k=5))
    out_path = osp.join('runs', outdir)
    os.makedirs(out_path)
    write_config(args, out_path)
    # ---
    data = Data(path=args.input, 
                template_path=args.template_struc, 
                N_DIVISIONS=args.n_divisions,
                test=args.test)
    train_size = round(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, (train_size, test_size))
    train_loader = DataLoader(train_data,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers=1)
    test_loader = DataLoader(test_data,
                             batch_size=64,
                             num_workers=1)
    # ---
    v_0, a_0, d_0, s_0 = data[0] # voxels, affinity, distance, score
    model = Model(*v_0.shape, 
                  dim=args.dim,
                  n_downsamples=args.downsamples,
                  n_recycles=args.recycles,
                  ) # init from shape
    # ---
    train(model, 
          train_loader, 
          learning_rate=args.learning_rate, 
          cuda=args.cuda, 
          epochs=args.epochs, 
          tensorboard=args.tensorboard,
          checkpoint=args.checkpoint,
          test_loader=test_loader,
          out_path=out_path,
          quiet=args.quiet) # doesn't do anything

    metrics = test(model, 
                   test_loader,
                   plot=False)

    sys.stdout.write(json.dumps(metrics))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-b','--batch_size',type=int,default=16)
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-6)
    parser.add_argument('-d','--dim',type=int,default=32)
    parser.add_argument('-y','--downsamples',type=int,default=6)
    parser.add_argument('-r','--recycles',type=int,default=6)
    parser.add_argument('-n','--n_divisions',type=int,default=16)
    # ---
    parser.add_argument('-i','--input',default='../outputs/newscore2-98k.csv')
    parser.add_argument('-o','--outdir')
    parser.add_argument('-t','--template_struc',default='../../../data/4KEY.pdb')
    parser.add_argument('-c','--cuda',action='store_true',default=False)
    parser.add_argument('-l','--tensorboard',action='store_true',default=False)
    parser.add_argument('-s','--checkpoint',action='store_true',default=False)
    parser.add_argument('-x','--test',action='store_true',default=False)
    parser.add_argument('-q','--quiet',action='store_true',default=False)
    args = parser.parse_args()
    main(args)
