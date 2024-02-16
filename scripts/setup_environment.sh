#!/usr/bin/env bash

PROJECT_DIR="$(realpath "$(dirname "$(dirname "$0")")")"

echo "Project directory: $PROJECT_DIR"

# load .env
if [ -f "$PROJECT_DIR"/config/.env ]; then
  source "$PROJECT_DIR"/config/.env
fi

apt-get update
apt-get install unzip wget nvtop -y

# github cli

# install
if ! command -v gh &> /dev/null
then
    type -p curl >/dev/null || (apt update && apt install curl -y)
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y
fi

# configure
echo "Configuring github..."
gh auth login --with-token < <(echo "$GITHUB_TOKEN")
gh auth setup-git

# rclone

# install
if ! command -v rclone &> /dev/null
then
    curl https://rclone.org/install.sh | bash
fi

# configure
"$(dirname "$0")/utils/config_rclone.sh"

# anaconda

# install
apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 libaio-dev

# curl miniforge to ./miniforge.sh
curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o "$PROJECT_DIR"/miniforge.sh

bash "$PROJECT_DIR"/miniforge.sh -b -p "$PROJECT_DIR"/conda
rm "$PROJECT_DIR"/miniforge.sh

# configure
source "$PROJECT_DIR"/conda/etc/profile.d/conda.sh
source "$PROJECT_DIR"/conda/etc/profile.d/mamba.sh

mamba init bash

mamba activate

pip install glances py3nvml

mamba create -n aidx-model-development
mamba env update -n aidx-model-development -f conda_environment.yml

apt-get clean -y
apt-get autoremove -y --purge
apt-get autoclean -y
