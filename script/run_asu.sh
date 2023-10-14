#!/bin/bash

# script for 4x4 ASU example 
# script all features added 

python ./cleanrl/ppo_continuous_action.py \
    --env-id control_y4 \
    --total-timesteps 5000000 \
    --num-envs 5 \
    --save-best-model True \
    --gamma 0.99 \
    --seed 1234 \
    --plot-logger True \
    --num-steps 4096 \

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
