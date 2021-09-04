#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Build Docker image and name it kaicd-efa-dti
docker build -t kaicd-efa-dti .
