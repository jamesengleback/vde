#!/bin/sh

USER_PASS=$1
# dependencies: git, curl, bzip2, gcc
apt update -y
apt upgrade -y
apt install git curl bzip2 gcc neovim tmux htop  -y

# --- user0 ---
adduser user --gecos "First Last,RoomNumber,WorkPhone,HomePhone" --disabled-password
echo user:$USER_PASS | chpasswd

cp ~/config/setupuser.sh /home/user
cp ~/config/.bashrc    /home/user
chmod 777 /home/user/setupuser.sh
