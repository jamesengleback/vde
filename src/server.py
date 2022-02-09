#!/home/u0/miniconda3/envs/enz/bin/python
import sys
import socket
import os
import os.path as osp
import json
import shutil

import enz
from utils import score_c

def evaluate(sequence,
             config,
             score_fn,
             ):
    protein = enz.Protein(config['template_struc'],
                          seq=sequence, 
                          keep=['HEM']) 

    protein.refold()

    docking_results = protein.dock(config['ligand'],
                                   target_sites=list(map(int, config['docking_site'])),
                                   exhaustiveness=int(config['exhaustiveness']))
    score = score_fn(protein, docking_results)
    return json.dumps({'score':score})

def main(args):
    if len(args) == 0:
        port = 8000
    else:
        port = int(args[0])
    with open('config.json') as f:
        config = json.load(f)
    print(f'localhost:{port}')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(('localhost',port))
        while True:
            msg_, addr = server.recvfrom(1024)
            msg = msg_.decode('utf-8')
            if msg == 'status':
                server.sendto('free'.encode(), addr)
            else:
                try:
                    data=evaluate(msg,config,score_fn=score_c)
                    print(data)
                    server.sendto(data.encode(), addr)
                except Exception as e:
                    server.sendto(repr(e), addr)
                    


if __name__ == '__main__':
    main(sys.argv[1:])
