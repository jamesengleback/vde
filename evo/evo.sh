#!/bin/bash
source ~/disk/src/miniconda3/etc/profile.d/conda.sh
conda activate enz
#for _ in $(seq  2); do
#        python evo.py -p 64 -e 4 -n 100 -s 0.25 1> /dev/null &
#done
seq 16 | parallel echo {} ; python evo.py -p 64 -e 4 -n 100 -s 0.25 1> /dev/null 
