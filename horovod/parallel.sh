#!/bin/bash

# Usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <training_script.py>"
    exit 1
fi

TRAINING_SCRIPT=$1

# Check parameter 
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: $TRAINING_SCRIPT not found!"
    exit 1
fi

# Run horovod.parallel
horovodrun -np 4 -H localhost:4 python "$TRAINING_SCRIPT"