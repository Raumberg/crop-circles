#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <training_script.py>"
    exit 1
fi

TRAINING_SCRIPT=$1

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: $TRAINING_SCRIPT not found!"
    exit 1
fi

horovodrun -np 4 -H localhost:4 python "$TRAINING_SCRIPT"
