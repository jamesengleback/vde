#!/bin/bash
# export USER_PASS
LABEL=$1
linode-cli linodes display $LABEL
IP=$(linode-cli linodes list  | grep $LABEL | awk '{print $14}')
scp -r config root@$IP:~
ssh root@$IP "~/config/setuproot.sh $USER_PASS"
ssh user@$IP '~/setupuser.sh'
ssh-copy-id user@$IP
