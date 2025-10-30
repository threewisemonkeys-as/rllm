export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout mk-k8s
git submodule update --init --recursive
pip install --user -e .
cd ..

git clone https://github.com/threewisemonkeys-as/verl.git
cd verl
git fetch 
git checkout main 
cd ..

pip install --user -e ./verl[vllm]
pip install --user -e .
