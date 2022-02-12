#!/bin/sh
ps aux | grep app.py | awk '{print $2}' | xargs kill
#HOSTS=$(grep http .active )
#PIDS=$(echo $HOSTS | awk '{print $2}')
#PORTS=$(ss -O | grep -E "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | sort | uniq)
## grep -E "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\:([a-zA-Z0-9]+)")
##echo $PORTS
## todo - check the process is vde
#for i in $PIDS; do
#	kill $i;
#done
