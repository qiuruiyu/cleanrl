#!/bin/bash

# script for 3x3 ASU example 
# script without part \mu item, which is the penalty for energy consumption. 

python ./cleanrl/ppo_continuous_action.py --env-id asu_no_mu --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1234

python ./cleanrl/ppo_continuous_action.py --env-id asu_no_mu --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1235

python ./cleanrl/ppo_continuous_action.py --env-id asu_no_mu --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1236

python ./cleanrl/ppo_continuous_action.py --env-id asu_no_mu --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1237

python ./cleanrl/ppo_continuous_action.py --env-id asu_no_mu --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1238