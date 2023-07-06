#!/bin/bash
# source /Users/joseph/miniforge3/etc/profile.d/conda.sh 
# conda activate ppo 

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 1 1 --num-envs 6 --total-timesteps 3000000  --track True --cuda False

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 1 5 --num-envs 6 --total-timesteps 3000000 --track True --cuda False  

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 1 10 --num-envs 6 --total-timesteps 3000000 --track True --cuda False 

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 1 20 --num-envs 6 --total-timesteps 3000000 --track True --cuda False 

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 1 30 --num-envs 6 --total-timesteps 3000000 --track True --cuda False 

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 0.1 1 --num-envs 6 --total-timesteps 3000000 --track True --cuda False 

python ./cleanrl/ppo_continuous_action.py --env-id Shell-v0 --qr 0.5 1 --num-envs 6 --total-timesteps 3000000 --track True --cuda False 
