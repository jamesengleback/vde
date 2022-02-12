#!/bin/bash
# environmental variable:
# ROOT_PASS
LABEL=$1

linode-cli linodes create \
	--type g6-dedicated-16 \
	--region eu-west \
	--label $LABEL \
	--image linode/debian11 \
	--root_pass $ROOT_PASS

ID=$(linode-cli linodes list  | grep $LABEL | awk '{print $2}')
IP=$(linode-cli linodes list  | grep $LABEL | awk '{print $14}')

echo "label $LABEL"            >   $LABEL-info
echo "ip $IP"                  >>  $LABEL-info
echo "id $ID"                  >>  $LABEL-info
echo "root_pass root@$LABEL"   >>  $LABEL-info
echo "user_pass user@$LABEL"   >>  $LABEL-info

if nc -z $IP 22 2>/dev/null; then
	ssh-copy-id root@$IP
fi

ssh-copy-id root@$IP
