#!/bin/sh

HITLIST=$@

for TGT in $HITLIST; do 
	echo "deleting $TGT"
	cat $TGT
	linode-cli linodes delete $(grep "id" $TGT | awk '{print $2}')

done

