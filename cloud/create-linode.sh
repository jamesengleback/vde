#!/bin/bash

pass generate -f root@evob > /dev/null
pass generate -f u0@evob > /dev/null
ROOT_PASS=$(pass root@evob)
U0_PASS=$(pass u0@evob)
LABEL=evob

linode-cli linodes create \
	--type g6-dedicated-50 \
	--region eu-west \
	--label $LABEL \
	--image linode/debian11 \
	--root_pass $ROOT_PASS

ID=$(linode-cli linodes list  | grep $LABEL | awk '{print $2}')
IP=$(linode-cli linodes list  | grep $LABEL | awk '{print $14}')

echo "label $LABEL"            >   $LABEL-info
echo "ip $IP"                  >>  $LABEL-info
echo "id $ID"                  >>  $LABEL-info
echo "root_pass root@evob"      >>  $LABEL-info
echo "u0_pass u0@evob"          >>  $LABEL-info

LOOP=true
while $LOOP; do
	sleep 5  ;
	STATUS=$(linode-cli linodes list | grep $LABEL | awk '{print $12}') ;
	echo $STATUS
	if  [[ $STATUS == *"running"* ]] ;
	then
		linode-cli linodes view $ID
		sshpass -p$ROOT_PASS scp -r config root@$IP:~
		sshpass -p$ROOT_PASS ssh root@$IP "~/config/setuproot.sh $U0_PASS"
		sshpass -p$U0_PASS  ssh u0@$IP '~/setupu0.sh'
		#sshpass -p$ROOT_PASS ssh-copy-id root@$IP
		#sshpass -p$U0_PASS ssh-copy-id u0@$IP
		break
	fi
done

#linode-cli linodes delete $ID

echo "alias evob='ssh root@$IP'" >> ~/.bashrc
