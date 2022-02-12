import sys
import requests
import json
from concurrent.futures import ThreadPoolExecutor

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

BM3_DM='MTIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFVGDGLVTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKVRK*'

if __name__ == '__main__':
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    helper = lambda p : get(seq=BM3_DM, 
                            addr=f"http://localhost:{p}", 
                            exhaustiveness=1)          
    with ThreadPoolExecutor() as pool:
        results = pool.map(helper, [5001,5002,5003,5004])
    o = list(results)
