#!/usr/bin/env python
import requests
import json
from ax.service.managed_loop import optimize
from ax.service.ax_client import AxClient

from api import API

def mutate_string(template, target_dict):
    s_ = list(template)
    for i,j in zip(target_dict.keys(), target_dict.values()):
        s_[i] = j
    return ''.join(s_)

BM3_DM='MTIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFVGDGLVTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKVRK*'



PARAMETERS = [{ "name": f"{i}", "type": "choice", "values" : list('ACDEFGHIKLMNPQRSTVWY')} \
                for i in [330, 75, 188,181] ]

def main(): 
    import sys
    api = API(sys.argv[1:])
    def helper(parameters):
        d = dict(zip(map(int, parameters.keys()), parameters.values()))
        seq = mutate_string(BM3_DM, d)
        score_ = api(seq)
        score = json.loads(score_['score'])
        return float(score['score'])

    client = AxClient()
    client.create_experiment(name='vde',parameters=PARAMETERS, objective_name='vde',minimize=True)
    for _ in range(8):
        params, idx = client.get_next_trial()
        client.complete_trial(trial_index=idx, raw_data=helper(params))

if __name__ == '__main__':
    main()

