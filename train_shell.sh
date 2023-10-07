#!/bin/bash
# source /Users/joseph/miniforge3/etc/profile.d/conda.sh 
# conda activate ppo 

python ./cleanrl/ppo_continuous_action.py --exp-name PPO --seed 1234 --cuda False --track False --save-model True --save-best-model True --save-interval 100 --plot-logger True --env-id Shell --qr 1 3 --num-envs 1 --total-timesteps 200000

python ./cleanrl/ppo_continuous_action.py --exp-name PPO --seed 1235 --cuda False --track False --save-model True --save-best-model True --save-interval 100 --plot-logger True --env-id Shell --qr 1 3 --num-envs 1 --total-timesteps 200000

python ./cleanrl/ppo_continuous_action.py --exp-name PPO --seed 1236 --cuda False --track False --save-model True --save-best-model True --save-interval 100 --plot-logger True --env-id Shell --qr 1 3 --num-envs 1 --total-timesteps 200000

python ./cleanrl/ppo_continuous_action.py --exp-name PPO --seed 1237 --cuda False --track False --save-model True --save-best-model True --save-interval 100 --plot-logger True --env-id Shell --qr 1 3 --num-envs 1 --total-timesteps 200000

python ./cleanrl/ppo_continuous_action.py --exp-name PPO --seed 1238 --cuda False --track False --save-model True --save-best-model True --save-interval 100 --plot-logger True --env-id Shell --qr 1 3 --num-envs 1 --total-timesteps 200000
