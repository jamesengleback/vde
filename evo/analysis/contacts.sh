#!/bin/bash

DIR="../runs"
cd ../runs
for RUN in $(ls); do
	for MUTANT in $(ls $RUN);
		do prody contacts -p $MUTANT/$RUN/$MUTANT/clean* $RUN/$MUTANT/mode*  ;
	done;
done

wait $(jobs -p)

cat *pdb | grep -v REMARK > ../analysis/all-contacts.pdb
rm *pdb
cd ../analysis
