import sys
import json
from flask import Flask
from flask import request

import enz
from utils import score_b

app = Flask(__name__)

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

@app.route('/', methods=['GET','PUT', 'POST'])
def main():
    record = json.loads(request.data)
    assert 'seq' in record.keys()
    seq = record['seq']
    return json.dumps(evaluate(seq,
                               config,
                               score_fn=score_b))

if __name__ == '__main__':
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    CONFIG = sys.argv[2]
    with open('config.json') as f:
        config = json.load(f)

    app.run(host='localhost',port=p)
