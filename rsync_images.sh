#!/bin/bash
# copy files from folders in data/img/
# to remote server
rsync -a --ignore-existing -e 'ssh -i ~/.ssh/MBP_Ireland.pem' --progress data/img/ ubuntu@ec2-54-229-54-8.eu-west-1.compute.amazonaws.com:/home/ubuntu/efs/data/img/