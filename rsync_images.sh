#!/bin/bash
# copy files from folders in data/img/
# to remote server
rsync -a --ignore-existing -e 'ssh -i ~/.ssh/MBP_Ireland.pem' --progress data/img/ingame/ ubuntu@ec2-54-154-110-123.eu-west-1.compute.amazonaws.com:/home/ubuntu/efs/data/img/ingame/