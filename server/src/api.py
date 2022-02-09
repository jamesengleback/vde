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
        return r.text
    else:
        return r.status_code

BM3_DM='MTIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFVGDGLVTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKVRK*'

if __name__ == '__main__':
    def helper(port): 
        return get(seq=BM3_DM, 
                   addr=f"http://127.0.0.1:{port}",
                   exhaustiveness=1)
    with ProcessPoolExecutor() as pool:
        results = pool.map(helper, range(5000, 5005))
    print([json.loads(i) for i in results if i is not None])
