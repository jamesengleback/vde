import torch
from torch import Tensor, LongTensor, cat, relu, sigmoid, zeros, randn, from_numpy
from torch.nn.functional import one_hot

import ga
import heapq
import random
from tqdm import tqdm

from simple_model import Data, AAS, Model, ResLin
import sys
sys.path.append('../..')
from bm3 import BM3_DM

SAA = dict(zip(AAS.values(), AAS.keys()))

def ohe(seq):
    return one_hot(LongTensor([AAS[i] for i in seq]), 
                   num_classes=len(AAS)).float()

def ham_str(a, b):
    return sum([i != j for i,j in zip(a,b)])

def main():
    POP_SIZE = 1024
    N_ITER = 8
    SURVIVAL = 0.25
    data = Data('../outputs/all-data-new2.csv', test=True)
    DM_GENE = ''.join([BM3_DM[i] for i in data.mutation_sites])
    X = ohe(DM_GENE)
    model = torch.load('simple-model.pt')

    tensor_item = lambda tensor : tensor.detach().cpu().item()


    def fitness(gene, dist_mean, aff_mean, dist_std, aff_std):
        ham = ham_str(gene, DM_GENE)
        ham_norm = ham / len(gene)
        score = sum([2*dist_mean, aff_mean, 2*ham_norm, -dist_std, -aff_std])
        return {'score':tensor_item(score),
                'dist_mean':tensor_item(dist_mean),
                'aff_mean':tensor_item(aff_mean),
                'dist_std':tensor_item(dist_std),
                'aff_std':tensor_item(aff_std),
                'ham':ham,
                'ham_norm':ham_norm,
                }

    mean = lambda l : sum(l) / len(l)
    o = {}
    pop = [ga.mutate(DM_GENE) for _ in range(POP_SIZE)]
    with tqdm(range(N_ITER)) as _iterations:
        for _ in _iterations:
            pop_ = cat([ohe(i).unsqueeze(0) for i in pop], dim=0)
            yh = cat([model(pop_).unsqueeze(-1) for _ in range(8)], dim=-1)
            yh_dist, yh_aff = yh[:,:,0], yh[:,:,1]
            yh_dist_mean, yh_aff_mean = yh_dist.mean(dim=-1), yh_aff.mean(dim=-1)
            yh_dist_std, yh_aff_std = yh_dist.std(dim=-1), yh_aff.std(dim=-1)
            pop__ = {i:fitness(i,j,k,l,m) for i,j,k,l,m in zip(pop, 
                                                           yh_dist_mean, 
                                                           yh_aff_mean,
                                                           yh_dist_std,
                                                           yh_aff_std,
                                                           )}
            best = heapq.nsmallest(round(POP_SIZE*SURVIVAL), 
                                   pop__, 
                                   key=lambda item : pop__[item]['score'])
            pop = [ga.mutate(ga.crossover(*random.choices(best, k=2))) \
                        for _ in range(POP_SIZE)]

            _iterations.set_postfix({
                        'mean_aff':mean([i['aff_mean'] for i in pop__.values()]),
                        'mean_dist':mean([i['dist_std'] for i in pop__.values()]),
                        'mean_score':mean([i['score'] for i in pop__.values()]),
                        'mean_ham':mean([i['ham'] for i in pop__.values()]),
                                    })
            o = {**o, **pop__}
    #print(DM_GENE)
    #from pprint import pprint
    #pprint(pop__)
    import pandas as pd
    df = pd.DataFrame(o).T
    print(df.sort_values('score', ascending=True)[:50])



if __name__ == '__main__':
    #main(int(sys.argv[1]))
    main()
