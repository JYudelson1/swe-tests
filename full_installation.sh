# apt -y install git-all
wget -qO- https://astral.sh/uv/install.sh | sh

apt-get -y install cuda-toolkit-12-4
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}} && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.4
apt-get -y install python3.10-dev

uv pip install torch
uv sync --no-install-package flash-attn
uv sync --no-build-isolation
