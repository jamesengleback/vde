#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate enz
export FLASK_APP=app.py
N=4
python app.py 5001
#for i in $(seq $N); do
#	python app.py 500$i &
#	pids[${i}]=$!
#done
#
#for pid in ${pids[*]}; do
#    wait $pid
#done
