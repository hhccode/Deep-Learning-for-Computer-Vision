#!/bin/bash
wget -O hw2_model.pkl 'https://www.dropbox.com/s/jq178kkkush8gkq/hw2_model.pkl?dl=1'
cp orig_models.py models.py
python3 hw2.py $1 $2