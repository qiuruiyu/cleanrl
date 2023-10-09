#!/bin/bash

python ./cleanrl/ppo_continuous_action.py --env-id asu33 --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1234

python ./cleanrl/ppo_continuous_action.py --env-id asu33 --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1235

python ./cleanrl/ppo_continuous_action.py --env-id asu33 --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1236

python ./cleanrl/ppo_continuous_action.py --env-id asu33 --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1237

python ./cleanrl/ppo_continuous_action.py --env-id asu33 --total-timesteps 3000000 --num-envs 5 --plot-logger True --save-best-model True --gamma 0.99 --seed 1238