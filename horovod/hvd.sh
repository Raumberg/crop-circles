#!/bin/bash

# Check parameter
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cpu|gpu>"
    exit 1
fi

# Install Horovod based on the parameter
if [ "$1" == "cpu" ]; then
    echo "Installing Horovod for CPU..."
    pip install horovod
elif [ "$1" == "gpu" ]; then
    echo "Installing Horovod for GPU with NCCL support..."
    HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
else
    echo "Invalid parameter. Please use 'cpu' or 'gpu'."
    exit 1
fi

echo "Horovod installation completed."