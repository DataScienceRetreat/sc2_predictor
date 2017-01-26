#!/bin/bash
# copy files from folders in data/img/
# to remote server
rsync -a --ignore-existing -e 'ssh -i ~/.ssh/MBP_Ireland.pem' --progress data/ingame/ ubuntu@ec2-54-154-110-123.eu-west-1.compute.amazonaws.com:/home/ubuntu/efs/data/ingame/
# efs mount instruction
# sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone).fs-16d428df.efs.eu-west-1.amazonaws.com:/ efs
