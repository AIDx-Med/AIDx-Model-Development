#!/usr/bin/env bash

SHELL=/bin/bash

source ./conda/etc/profile.d/conda.sh
source ./conda/etc/profile.d/mamba.sh

mamba activate aidx-model-development

jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 --ServerApp.token='' --ServerApp.password=''