#!/usr/bin/env python
import sys
import os 
from itertools import cycle
import json
import requests
import asyncio

from flask import Flask
from flask import request


class Pod:
    def __init__(self, 
                port,
                config,
                *args):
        self.port = port
        self.config = config
        self.url = f'http://localhost:{self.port}'
        self.proc = os.popen(f'python pod.py {self.port} {self.config}')
        self.log = []
    def __call__(self, seq):
        if self.alive and not self.busy:
            r = requests.get(self.url, 
                             data=json.dumps({'seq':seq}))
            if r.status_code == 200:
                return json.loads(r.text)
            else:
                return r.status_code
    @property
    def busy(self):
        return False
    @property
    def alive(self):
        return True

class Manager:
    def __init__(self, 
                 n_process,
                 config,
                 *args):
        self.n_process = n_process
        self.config = config
        self.rundir = 'runs'
        self.pods = cycle([Pod(5001+i, config) for i in range(n_process)])
        self.jobq = []

    def __call__(self, seq):
        return next(self.pods)(seq)
    def checkalive(self):
        alive = [psutil.pid_exists(p._proc.pid) \
                        for p in self.processes]
        return dict(zip(self.processes, alive))

    def top(self):
        return {'cpu%':psutil.cpu_percent,
                'du':None}



if __name__ == '__main__':
    if len(sys.argv) == 1:
        n_process = 1
    else:
        n_process = int(sys.argv[1])
    config = 'config.json'

    manager = Manager(n_process, config='config.json')

    app = Flask(__name__)

    @app.route('/', methods=['GET','PUT', 'POST'])
    def main():
        record = json.loads(request.data)
        assert 'seq' in record.keys()
        seq = record['seq']
        return json.dumps(manager(seq))

    app.run()


