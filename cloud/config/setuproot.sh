#!/bin/sh

U0_PASS=$1
# dependencies: git, curl, bzip2, gcc
apt update -y
apt upgrade -y
apt install git curl bzip2 gcc neovim tmux htop  -y

# --- user0 ---
adduser u0 --gecos "First Last,RoomNumber,WorkPhone,HomePhone" --disabled-password
echo u0:$U0_PASS | chpasswd

cp ~/config/setupu0.sh /home/u0
cp ~/config/.bashrc    /home/u0
chmod 777 /home/u0/setupu0.sh
