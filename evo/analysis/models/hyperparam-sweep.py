# just a random search.
#
import sys
import os
import random
from ast import literal_eval
from dataclasses import dataclass
import json
from pprint import pprint
import argparse
import pandas as pd

@dataclass
class ParamSpace:
    epochs = [5,10,20,50]
    batch_size = [32, 64, 128, 256]
    learning_rate = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    dim = [2**i for i in range(5,9)]
    downsamples = [2,4,6]
    recyles = [2,4,6,8]


def main(num):
    param_space = ParamSpace()
    batch = [
            {'epochs':random.choice(param_space.epochs),
             'batch_size':random.choice(param_space.batch_size),
             'learning_rate':random.choice(param_space.learning_rate),
             'dim':random.choice(param_space.dim),
             'downsamples':random.choice(param_space.downsamples),
             'recyles':random.choice(param_space.recyles)
             }
             for _ in range(args.num_rums)]


    data = []
    for params in batch:
        cmd = f"python train.py  "\
            + f" -e {params['epochs']}"\
            + f" -b {params['batch_size']}"\
            + f" -lr {params['learning_rate']}"\
            + f" -d {params['dim']}"\
            + f" -y {params['downsamples']}"\
            + f" -r {params['recyles']}"
        if args.checkpoint:
            cmd += " -s"
        if args.tensorboard:
            cmd += " -l"
        if args.test:
            cmd += " -x"
        if args.cuda:
            cmd += " -c"
        if args.device is not None:
            cmd += f" --device {args.device}"
        if args.quiet:
            cmd += " 2>/dev/null"
        subprocess = os.popen(cmd)
        output = subprocess.read()
        try:
            metrics = {**json.loads(output), **{'fail':False}}
        except:
            #metrics = {'loss_aff':'nan', 'loss_dist':'nan', 'loss_score':'nan'} | {'fail':True}
            metrics = {**{'loss_aff':'nan', 'loss_dist':'nan', 'loss_score':'nan'}, **{'fail':True}}
        data.append({**params, **metrics})
    df = pd.DataFrame(data)
    df.to_csv(sys.stdout, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--checkpoint',action='store_true',default=False)
    parser.add_argument('-l', '--tensorboard',action='store_true',default=False)
    parser.add_argument('-x', '--test',action='store_true',default=False)
    parser.add_argument('-c', '--cuda',action='store_true',default=False)
    parser.add_argument('-d', '--device')
    parser.add_argument('-q', '--quiet',action='store_true',default=False)
    parser.add_argument('-n', '--num_rums', type=int, default=5)
    args = parser.parse_args()
    main(args)
