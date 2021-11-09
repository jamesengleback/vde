# just a random search.
#
import sys
import os
import random
from dataclasses import dataclass
import json

import pandas as pd

@dataclass
class ParamSpace:
    epochs = [5,10,20,50]
    batch_size = [8, 16, 32]
    learning_rate = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    dim = [2**i for i in range(3,8)]
    downsamples = [2,4,6,8]
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
             for _ in range(num)]


    output = []
    for params in batch:
        subprocess = os.popen(f"python train.py  "\
                            + f" -e {params['epochs']}"\
                            + f" -b {params['batch_size']}"\
                            + f" -lr {params['learning_rate']}"\
                            + f" -d {params['dim']}"\
                            + f" -y {params['downsamples']}"\
                            + f" -r {params['recyles']}"\
                            + f" -s"\
                            + f" -l"\
                            + f" -c"\
                            #+ f" -x"\
                            #+ f" 2>/dev/null"\
                            )
        try:
            #metrics = json.loads(subprocess.read()) | {'fail':False}
            metrics = json.loads({**subprocess.read(), **{'fail':False}})
        except:
            #metrics = {'loss_aff':'nan', 'loss_dist':'nan', 'loss_score':'nan'} | {'fail':True}
            metrics = {**{'loss_aff':'nan', 'loss_dist':'nan', 'loss_score':'nan'}, **{'fail':True}}
        output.append({**params, **metrics})
    df = pd.DataFrame(output)
    df.to_csv(sys.stdout)


if __name__ == '__main__':
    main(int(sys.argv[1]))
