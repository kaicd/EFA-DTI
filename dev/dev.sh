#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Build image
./dev/build.sh

# Start the container with a bash session
# Everything done in the container will be gone after
# you close the terminal except changes to your current working directory.
# Your current working directory will be mounted into the container which
# should help making changes to the code and try it out. Changes to anything
# outside of your current working directory will not be persistent.


# Note: https://github.com/pytorch/pytorch#docker-image
# Parameters:
# --rm          => Delete container when it stops
# -it           => Make container interactive
# --gpus all    => Use all GPUs
# --ipc host    => Use the host system’s IPC namespace for pytorch
# -v some:some  => Mount storage volume, wandb and code
docker run \
    --rm \
    -it \
    --gpus=all \
    --ipc=host \
    -v $(pwd)/storage:/raid/KAICD_sarscov2/efa_dti \
    -v $(pwd)/wandb:/app/wandb \
    -v $(pwd):/app \
    /bin/bash
