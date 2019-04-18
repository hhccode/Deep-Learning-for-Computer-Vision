#!/bin/bash
wget -O hw2_best_model.pkl 'https://www.dropbox.com/s/b6cfzhww4tz1on8/hw2_bestmodel.pkl?dl=1'
cp improve_models.py models.py
python3 hw2_best.py $1 $2