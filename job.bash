#!/bin/bash
ARGS="$1 $2"
cd /home/semai/Software/dubins
echo $ARGS
source bin/activate
python experiment.py $ARGS
deactivate
