#!/bin/bash
#source ~/miniconda3/etc/profile.d/conda.sh
source ~/src/miniconda/miniconda3/etc/profile.d/conda.sh

conda activate enz
export FLASK_APP=app.py
date > .active
for i in $(seq $1); do
	PORT=500$i
	python app.py $PORT &
	pids[${i}]=$!
	echo http://localhost:$PORT $! >> .active
done

for pid in ${pids[*]}; do
    wait $pid
done
