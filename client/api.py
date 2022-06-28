#!/usr/bin/env python
from itertools import cycle
import requests
import json
from concurrent.futures import ProcessPoolExecutor

def get(seq,
        addr,
        **kwargs,
        ):
    d = {'seq':seq}
    #d = {**d, **kwargs}
    r = requests.get(addr, data=json.dumps(d))
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        return r.status_code

class API:
    def __init__(self, *args):
        self.addresses = cycle(*args)
    def __call__(self, seq, **kwargs):
        return get(seq, next(self.addresses), **kwargs)


BM3_DM='MTIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFVGDGLVTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKVRK*'

if __name__ == '__main__':
    import sys
    addresses = sys.argv[1:]
    api = API(addresses)
    with ProcessPoolExecutor() as pool:
        results = pool.map(api, [BM3_DM] * len(addresses))
    print(list(results))
