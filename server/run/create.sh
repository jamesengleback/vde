#!/bin/bash
# environmental variable:
# ROOT_PASS
LABEL=$1


linode-cli linodes create \
	--type g6-dedicated-4 \
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
#LOOP=true
#while $LOOP; do
#	sleep 5  ;
#	STATUS=$(linode-cli linodes list | grep $LABEL | awk '{print $12}') ;
#	echo $STATUS
#	if  [[ $STATUS == *"running"* ]] ;
#	then
#		linode-cli linodes view $ID
#		sshpass -p$ROOT_PASS scp -r config root@$IP:~
#		sshpass -p$ROOT_PASS ssh root@$IP "~/config/setuproot.sh $U0_PASS"
#		sshpass -p$USER_PASS  ssh user@$IP '~/setupu0.sh'
#		break
#	fi
#done

ssh-copy-id root@$IP
