#!/bin/bash

DIR="../runs"
for RUN in $(ls $DIR); do
	for MUTANT in $(ls $DIR/$RUN);
		do prody contacts -p $DIR/$MUTANT $DIR/$RUN/$MUTANT/clean* $DIR/$RUN/$MUTANT/mode* ;
	done;
done

cat *pdb | grep -v REMARK > ../analysis/all-contacts.pdb
rm *pdb
cd ../analysis
