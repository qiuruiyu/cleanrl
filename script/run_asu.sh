#!/bin/bash

# script for 4x4 ASU example 
# script all features added 

# rm -rf ./runs/ppo_continuous_action_control_ai701*
# rm -rf ./models/ppo_continuous_action_control_ai701*
rm -rf ./runs/figs/

python ./cleanrl/ppo_continuous_action.py \
    --env-id control_asu4 \
    --total-timesteps 10000000 \
    --num-envs 5 \
    --save-best-model True \
    --gamma 0.99 \
    --seed 1234 \
    --num-steps 4096 \
    --plot-logger True \
    

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id control_y4 \
#     --total-timesteps 5000000 \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1235 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id control_y4 \
#     --total-timesteps 5000000 \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1236 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id control_y4 \
#     --total-timesteps 5000000 \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1237 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id control_y4 \
#     --total-timesteps 5000000 \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1238 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id control_y4 \
#     --total-timesteps 5000000 \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1239 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id control_y4 \
#     --total-timesteps 5000000 \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1240 \
#     --plot-logger True \
