#!/bin/bash

# conda doesn't support sh
# dependencies: git, curl, bzip2, gcc

sudo apt install git curl bzip2 gcc

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
	./$FNAME
	~/miniconda3/bin/conda init
source ~/.bashrc

# --- linode ---
cd ~
pip install linode-cli
linode-cli obj get james-engleback PyRosetta4.MinSizeRel.python38.ubuntu.release-284.tar.bz2

# --- enz ---
cd ~/src
git clone https://github.com/jamesengleback/enz
cd enz
conda env create -f env.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate enz

else
	echo $miniconda.urls
	echo $DOWNLOAD
fi
# --- pyrosetta ---
cd ~
tar xfvj PyRosetta4.MinSizeRel.python38.ubuntu.release-284.tar.bz2
pip install ~/PyRosetta4.MinSizeRel.python38.ubuntu.release-284/setup

# --- enz again ---
pip install ~/src/enz

# --- evo ---
git clone https://github.com/jamesengleback/evo
cd evo

