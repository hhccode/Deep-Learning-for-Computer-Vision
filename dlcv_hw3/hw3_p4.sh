#!/bin/bash
if [ "$2" = "mnistm" ] || [ "$2" = "usps" ]
then
    python3 gta.py $1 $2 $3
elif [ "$2" = "svhn" ]
then
    python3 adda.py $1 $2 $3
fi