import sys
import json
from flask import Flask
from flask import request
from server import evaluate
from utils import score_c

app = Flask(__name__)

with open('config.json') as f:
    config = json.load(f)

@app.route('/', methods=['GET','PUT', 'POST'])
def main():
    record = json.loads(request.data)
    assert 'seq' in record.keys()
    seq = record['seq']
    return json.dumps({'score':evaluate(seq,
                                        config,
                                        score_fn=score_c)})

if __name__ == '__main__':
    port = int(sys.argv[1])
    app.run(host='localhost',port=port)
