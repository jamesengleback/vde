#!/bin/bash

# --- conda ---
mkdir -p src/miniconda
cd src/miniconda
PYTHON_VERSION="py38"
CPU_ARCH=$(uname -m)
URLS=$(curl https://docs.conda.io/en/latest/miniconda.html | grep -E -o "https.*\.sh")
echo "# minicoda urls for $CPU_ARCH" > miniconda.urls
for i in $URLS; do
	echo $i | grep -i linux | grep $CPU_ARCH >> miniconda.urls
done
DOWNLOAD=$(grep $PYTHON_VERSION miniconda.urls)
if  [ $(echo $DOWNLOAD | wc -l) -lt 2 ]
then
	FNAME=$(echo $DOWNLOAD | cut -d/ -f5)
	curl $DOWNLOAD > $FNAME
	chmod +x $FNAME
	./$FNAME -b
	~/miniconda3/bin/conda init
source ~/.bashrc

# --- enz ---
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/src
git clone https://github.com/jamesengleback/enz
git clone https://github.com/jamesengleback/ga
cd enz
conda env create -f env.yml
conda activate enz

else
	echo $miniconda.urls
	echo $DOWNLOAD
fi

# --- linode ---
cd ~/src
pip install boto
pip install linode-cli

##### get token!
linode-cli obj get james-engleback PyRosetta4.MinSizeRel.python38.ubuntu.release-284.tar.bz2

# --- pyrosetta ---
cd ~
tar xfvj PyRosetta4.MinSizeRel.python38.ubuntu.release-284.tar.bz2
pip install ~/src/PyRosetta4.MinSizeRel.python38.ubuntu.release-284/setup


pip install ~/src/enz
pip install ~/src/ga

cd ~
git clone https://github.com/jamesengleback/evo
pip install ~/evo
