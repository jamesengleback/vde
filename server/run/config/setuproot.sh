#!/bin/sh
sed -i 's/PasswordAuthentication no/PasswordAuthentication no/g' /etc/ssh/sshd_config
sed -i 's/#MaxAuthTries 6/MaxAuthTries 6/g' /etc/ssh/sshd_config

USER_PASS=$1
# dependencies: git, curl, bzip2, gcc
apt update -y
apt upgrade -y
apt install git curl bzip2 gcc  -y
apt install neovim tmux htop  -y # cc

# --- user ---
adduser user --gecos "First Last,RoomNumber,WorkPhone,HomePhone" --disabled-password
passwd user

cp ~/config/setupuser.sh /home/user
cp ~/config/.bashrc    /home/user
chmod 777 /home/user/setupuser.sh
