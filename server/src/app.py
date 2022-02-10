#!/usr/bin/env python
import sys
import os 
import psutil
import requests

from flask import Flask
from flask import request

if len(sys.argv) == 1:
    n_process = 1
else:
    n_process = int(sys.argv[1])

app = Flask(__name__)

class Manager:
    def __init__(self, 
                 n_process,
                 *args):
        self.n_process = n_process
        self.processes = [os.popen(f'python app.py {5000+i}') \
                          for i in range(n_process)]
        self.rundir = 'runs'
    def checkalive(self):
        alive = [psutil.pid_exists(p._proc.pid) \
                        for p in self.processes]
        return dict(zip(self.processes, alive))
    def top(self):
        return {'cpu%':psutil.cpu_percent,
                'du':None}


@app.route('/')
def eval():
    pass

manager = Manager(n_process)
app.run()

