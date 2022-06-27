#!/bin/bash

pass generate -f root@analysis > /dev/null
pass generate -f u0@analysis > /dev/null
ROOT_PASS=$(pass root@analysis)
U0_PASS=$(pass u0@analysis)
LABEL=analysis

linode-cli linodes create \
	--type g6-dedicated-8 \
	--region eu-west \
	--label $LABEL \
	--image linode/debian11 \
	--root_pass $ROOT_PASS

ID=$(linode-cli linodes list  | grep $LABEL | awk '{print $2}')
IP=$(linode-cli linodes list  | grep $LABEL | awk '{print $14}')

echo "label $LABEL"            >   $LABEL-info
echo "ip $IP"                  >>  $LABEL-info
echo "id $ID"                  >>  $LABEL-info
echo "root_pass root@analysis"      >>  $LABEL-info
echo "u0_pass u0@analysis"          >>  $LABEL-info

while true; do
	sleep 1  ;
	STATUS=$(linode-cli linodes list | grep $LABEL | awk '{print $12}') ;
	echo "$(date) $STATUS"
	if  [[ $STATUS == *"running"* ]] ;
	then
		break
	fi
done

linode-cli linodes view $ID
sshpass -p$ROOT_PASS scp -r config root@$IP:~
sshpass -p$ROOT_PASS ssh root@$IP "~/config/setuproot.sh $U0_PASS"
sshpass -p$U0_PASS  ssh u0@$IP '~/setupu0.sh'
sshpass -p$ROOT_PASS ssh-copy-id root@$IP
sshpass -p$U0_PASS ssh-copy-id u0@$IP
#linode-cli linodes delete $ID

echo "alias analysis='ssh root@$IP'" >> ~/.bashrc

