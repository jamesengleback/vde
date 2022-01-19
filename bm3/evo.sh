#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate enz
for _ in $(seq  8); do
        python main.py -b -p 128 -e 16 -n 32 -s 0.25  &
done
#tar cfz runs.tar.gz runs/
#conda deactivate
#linode-cli obj put runs.tar.gz james-engleback

