#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Build image
./dev/build.sh

# Run the training
# Mount the data directory to the location configured in:
# config/efa_dti_config.yaml
# Mount wandb directory to be persistet after the process exits

# Note: https://github.com/pytorch/pytorch#docker-image
# Parameters:
# --rm          => Delete container when it stops
# -it           => Make container interactive
# --gpus all    => Use all GPUs
# --ipc=host    => Use the host system’s IPC namespace for pytorch
# -v some:some  => Mount storage volume, wandb and code
docker run \
    --rm \
    -it \
    --gpus all \
    --ipc=host \
    -v $(pwd)/storage:/raid/KAICD_sarscov2/efa_dti \
    -v $(pwd)/wandb:/app/wandb \
    kaicd-efa-dti
