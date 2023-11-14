#!/bin/bash

# script for 4x4 ASU example 
# script all features added 

# --track True \
# --wandb-project-name ASU \

rm -rf ./runs/figs/

python ./cleanrl/ppo_continuous_action.py \
    --env-id stable4x4 \
    --total-timesteps 7000000 \
    --num-envs 5 \
    --save-best-model True \
    --gamma 0.99 \
    --seed 43124 \
    --num-steps 8192 \
    --plot-logger True \
        

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id stable4x4 \
#     --total-timesteps 7000000 \
#     --track True \
#     --wandb-project-name ASU \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 4321 \
#     --num-steps 8192 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id stable4x4 \
#     --total-timesteps 7000000 \
#     --track True \
#     --wandb-project-name ASU \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 6522 \
#     --num-steps 8192 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id stable4x4 \
#     --total-timesteps 7000000 \
#     --track True \
#     --wandb-project-name ASU \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 1237 \
#     --num-steps 8192 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id stable4x4 \
#     --total-timesteps 7000000 \
#     --track True \
#     --wandb-project-name ASU \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 6523 \
#     --num-steps 8192 \
#     --plot-logger True \

# python ./cleanrl/ppo_continuous_action.py \
#     --env-id stable4x4 \
#     --total-timesteps 7000000 \
#     --track True \
#     --wandb-project-name ASU \
#     --num-envs 5 \
#     --save-best-model True \
#     --gamma 0.99 \
#     --seed 3214 \
#     --num-steps 8192 \
#     --plot-logger True \
