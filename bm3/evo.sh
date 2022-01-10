#!/bin/bash
source ~/disk/src/miniconda3/etc/profile.d/conda.sh
conda activate enz
for _ in $(seq  2); do
        python main.py -p 64 -e 4 -n 10 -s 0.25 1> /dev/null &
done
#seq 8 | parallel echo {} ; python main.py -p 64 -e 8 -n 20 -s 0.25 1> /dev/null 
#python main.py -p 16 -e 1 -n 8
