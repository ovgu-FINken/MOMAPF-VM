#!/bin/bash
cd /home/semai/Software/dubins
echo $(pwd)
source bin/activate
python experiment.py --fetch
deactivate
