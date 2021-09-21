#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Sort imports
python -m isort efa_dti
python -m isort utility

# Format code
python -m black -q efa_dti
python -m black -q utility

# Format docstrings
python -m docformatter -i -r efa_dti
python -m docformatter -i -r utility
