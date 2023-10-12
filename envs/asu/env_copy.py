import copy
import os 
import sys
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.wrappers import (NormalizeObservation, NormalizeReward, TimeAwareObservation)
from scipy import io


class ASUEnv(gym.Env):
    def __init__(self,
                 env_config: dict
                 ) -> None:
        """
        System initialization, 7x7 system 
        """
        self.Tsim = env_config['Tsim'] if 'Tsim' in env_config.keys() else 2500
        self.obs_dict = env_config['obs_dict'] if 'obs_dict' in env_config.keys() else False
        self.nu = 3
        self.ny = 3
        self.back = 10
        # self.back = self.Tsim
        self.total_reward = []
        self.u0 = np.array([
            93285,  # 空气调节
            56.466, # 氧气调节
            47.4,   # 氮气调节
            57.34,  # V3调节
            # 93285,  # 空气量
            # 17940,  # 氧气量
            # 37672,  # 氮气量
        ]).reshape(-1, 1)
        self.y0 = np.array([
            93285,    # 空气量
            17940,    # 氧气量
            37672,    # 氮气量
            1.434,    # 污氮气含量
            99.7648,  # 氧气纯度 
            0.317,    # 氮气纯度
            9.55,     # AI701 * 
        ]).reshape(-1, 1)

        self.goal = np.array([
            99000, 
            19000,
            39976, 
            1.57, 
            99.7332, 
            0.3508, 
            9.3461
        ]).reshape(-1, 1)
        
        self.u0 = self.u0[:self.nu].reshape(-1, 1)
        self.y0 = self.y0[:self.ny].reshape(-1, 1)
        self.goal = self.goal[:self.ny].reshape(-1, 1)

        self.e0_real = self.goal - self.y0  # init real error 
        self.e0 = (self.goal - self.y0) / self.e0_real  # init scaled error 
        
        self.target = np.zeros((self.ny, 1))
        self.epsilon = 0   # relaxable error
        self.relaxed_target = self.epsilon * np.repeat(-self.epsilon * np.ones((self.ny, 1)) * (self.e0 > 0), self.back, axis=1)
        
        # weight parameters 
        self.Q = np.ones((1, self.ny))
        self.R = np.ones((1, self.nu))

        # step parameters 
        self.S = io.loadmat('./envs/asu/step_4.mat')['Step']
        # self.S = io.loadmat('./envs/asu/S4.mat')['S']
        self.S = self.S[:self.Tsim, :, :]

        '''
        =================== action space =================
        | du1 | du2  | du3  | du4  | du5 |  du6  |  du7  | 
        |  0  |   0  |  0   |  0   |  0  |  0    |  0    |
        |  6  | 4e-3 | 1e-2 | 2e-3 | 3e3 |  600  |  1200 |
        ==================================================
        '''
        if self.nu == 4:
            self.action_high = np.array([6, 4e-3, 1e-2, 2e-3]).astype(np.float32) * 3
        elif self.nu == 3:
            self.action_high = np.array([6, 4e-3, 1e-2]).astype(np.float32) * 3
        else:
            raise NotImplementedError
        
        self.action_low = -self.action_high

        self.action_space = spaces.Box(
            -1, 1,
            shape=(self.nu,)
        )

        '''
        ================= observation space ================ 
        | e | 
        | u | 
        ====================================================
        '''
        if self.obs_dict:
            self.observation_space = spaces.Dict(
                spaces={
                    "error": spaces.Box(-np.inf, np.inf, (self.ny, self.back), dtype=np.float32),
                    "u": spaces.Box(-10, 10, (self.nu, self.back), dtype=np.float64),
                } 
            )
        else: 
            self.observation_space = spaces.Box(
                -np.inf, np.inf,
                shape=(2*self.ny+self.nu, self.back)
            )

        self.state = self.assign_init_state() 

    def assign_init_state(self):
        error = np.repeat(self.e0, self.back, axis=1)
        y = np.repeat(self.e0, self.back, axis=1)
        du = np.zeros((self.nu, self.back))
        u = np.repeat(np.ones((self.nu, 1)), self.back, axis=1)

        if self.obs_dict:
            init_state = {
                "error": error,
                "u": u,
            }
        else:  # array like 
            init_state = np.vstack(
                (
                    self.relaxed_target, 
                    error,
                    u,
                )
            )
        return init_state

    def get_obs_info(self, obs):
        if self.obs_dict:
            obs_info = [obs[key] for key in obs.keys()]
            return tuple(obs_info)
        else:
            error_ = obs[self.ny:2*self.ny, :]
            u_ = obs[-self.nu:, :]
            return tuple([error_, u_])
    
    def update_state(self, info:tuple):
        '''
        tuple format like (current_error, action)
        '''
        error_, u_ = self.get_obs_info(self.state)
        current_error, current_u = info
        # current_u start from 1
        current_u /= self.u0

        '''
        hstack current state value
        '''
        error_ = np.hstack((
            error_[:, 1:], current_error
        ))

        u_ = np.hstack((
            u_[:, 1:], current_u
        ))
        
        '''
        vtstack the time frame for state value 
        '''
        if self.obs_dict:
            self.state = {
                "error": error_,
                "u": u_,
            }
        else:
            self.state = np.vstack((
                self.relaxed_target, error_, u_,
            ))
    
    def rescale_action(self, action):
        # [-1, 1]
        action = action.squeeze() # change to vector
        res = action * self.action_high 
        return res
    
    def calculation_output(self):
        # y1 - y3
        delta = (self.usim[1:self.num_step+1, :3] - self.usim[:self.num_step, :3]).T
        step_cut = np.vstack(
            (
                np.flipud(self.S[:self.num_step, 0, 0]).reshape(1, -1),
                np.flipud(self.S[:self.num_step, 1, 1]).reshape(1, -1),
                np.flipud(self.S[:self.num_step, 2, 2]).reshape(1, -1),
            )
        )
        change = np.sum(delta * step_cut, axis=-1).reshape(-1, 1) 
        # update 
        self.ysim[self.num_step, :3] = self.ysim[0, :3] + change.reshape(1, -1)
        
        # delta is same for all y later 
        if self.ny >= 4:
            delta = np.vstack(
                (
                    (self.usim[1:self.num_step+1, 3] - self.usim[:self.num_step, 3]).reshape(1, -1),
                    (self.ysim[1:self.num_step+1, 0] - self.ysim[:self.num_step, 0]).reshape(1, -1),
                    (self.ysim[1:self.num_step+1, 1] - self.ysim[:self.num_step, 1]).reshape(1, -1),
                    (self.ysim[1:self.num_step+1, 2] - self.ysim[:self.num_step, 2]).reshape(1, -1),
                )
            )

        if self.ny >= 4:
            # y4 (AI5)
            step_cut = np.vstack(
                (
                    np.flipud(self.S[:self.num_step, 3, 3]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 3, 4]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 3, 5]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 3, 6]).reshape(1, -1),
                )
            )
            change = np.sum(delta * step_cut)
            # update 
            self.ysim[self.num_step, 3] = self.ysim[0, 3] + change

        if self.ny >= 5:
            # y5 (AIAS102)
            step_cut = np.vstack(
                (
                    np.flipud(self.S[:self.num_step, 4, 3]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 4, 4]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 4, 5]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 4, 6]).reshape(1, -1),
                )
            )
            change = np.sum(delta * step_cut)
            # update 
            self.ysim[self.num_step, 4] = self.ysim[0, 4] + change
            if self.ysim[self.num_step, 4] >= 99.98:
                self.ysim[self.num_step, 4] = 99.98

        if self.ny >= 6:
            # y6 (ASAS103)
            step_cut = np.vstack(
                (
                    np.flipud(self.S[:self.num_step, 5, 3]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 5, 4]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 5, 5]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 5, 6]).reshape(1, -1),
                )
            )
            change = np.sum(delta * step_cut)
            # update 
            self.ysim[self.num_step, 5] = self.ysim[0, 5] + change

        if self.ny >= 7:
            # y7 (AI701)
            step_cut = np.vstack(
                (
                    np.flipud(self.S[:self.num_step, 6, 3]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 6, 4]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 6, 5]).reshape(1, -1),
                    np.flipud(self.S[:self.num_step, 6, 6]).reshape(1, -1),
                )
            )
            change = np.sum(delta * step_cut)
            # update 
            self.ysim[self.num_step, 6] = self.ysim[0, 6] + change 

    def render_plot(self):
        t = np.array(range(self.Tsim))
        fig, axs = plt.subplots(np.max([self.ny, self.nu]), 2, figsize=(12, 9))
        # ny | nu 
        for i in range(self.ny):
            axs[i, 0].plot(t, self.ysim[:-1, i], 'b-', label='y{}'.format(i+1))
            # if i < 3:
            axs[i, 0].plot([0, t[-1]], [self.goal[i], self.goal[i]], 'r--', label='goal')
            # else:
            #     axs[i, 0].plot([0, t[-1]], [self.goal[i]-0.6*(self.e0_real[i]), self.goal[i]-0.6*(self.e0_real[i])], 'r--', label='goal')
            #     axs[i, 0].plot([0, t[-1]], [self.goal[i]-0.8*(self.e0_real[i]), self.goal[i]-0.8*(self.e0_real[i])], 'r--', label='goal')
            axs[i, 0].grid(True)
            axs[i, 0].legend()
        for i in range(self.nu):
            axs[i, 1].plot(t, self.usim[:-1, i], 'b-', label='u{}'.format(i+1))
            axs[i, 1].grid(True)
            axs[i, 1].legend()
        plt.savefig('runs/figs/fig_{}_{}.png'.format(len(self.total_reward), int(self.total_reward[-1].item())))

    def step(self, action):
        """
        :param action: 
        :return: tuple(observation, reward, terminated, truncated, info)
        """
        action = action.reshape(-1, 1)
        self.num_step += 1
        self.du = self.rescale_action(action).reshape(-1, 1)
        # self.du = action
        self.u += self.du
        self.dusim[self.num_step, :] = self.du.squeeze() 
        self.usim[self.num_step, :] = self.u.squeeze() 

        self.calculation_output()

        current_y = self.ysim[self.num_step, :].reshape(-1, 1)
        current_error = (self.goal - current_y) / self.e0_real
        self.update_state((current_error, self.u.copy()))

        reward = 0 
        terminated = False
        truncated = False

        """Terminated and Truncated Judgement"""
        if self.num_step == self.Tsim:
            truncated = True
            if len(self.total_reward) % 30 == 1:
                os.makedirs('runs/figs', exist_ok=True)
                self.render_plot()

        """Reward Design"""
        # mse error of 3 deterministic vars in y 
        reward -= np.sum((current_error[:4] - self.epsilon)**2)
        
        # consumption of 3 vars in u, action is [-1, 1]
        reward -= np.sum(action**2) * 0.015

        # prevent over shooting
        for i in range(3):  # only for the 3 vars ahead
            if self.e0[i] * current_error[i] < 0:  # over shooting 
                reward -= np.log(self.num_step) * np.abs(current_error[i])

        self.total_reward[-1] += reward 

        return self.state, reward, terminated, truncated, {} 

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            super().reset(seed=seed)    

        self.num_step = 0
        self.total_reward.append(0)
        # reset monitor params 
        self.du = np.zeros((self.nu, 1))
        self.u = self.u0.copy() 
        self.y = self.y0.copy()
        self.dusim = np.zeros((self.Tsim+1, self.nu))
        self.usim = np.vstack((
            self.u0.copy().T,
            np.zeros((self.Tsim, self.nu))
        ))
        self.ysim = np.vstack((
            self.y0.copy().T,
            np.zeros((self.Tsim, self.ny))
        ))         

        self.state = self.assign_init_state()

        return self.state, {} 

def make_asu_env(rank=0, seed=0):
    env = ASUEnv(
        {
            'Tsim': 1500,
            'obs_dict': False
        }
    )
    env.reset(seed=seed+rank)
    return env


if __name__ == "__main__":
    # num_cpu = 15
    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], 'spawn')
    # env = VecNormalize(env, norm_obs=False)
    eval_env = ASUEnv(
        {
            'Tsim': 1500,
            'obs_dict': False
        }
    )

    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, DDPG, SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    # agent = PPO(
    #     policy='MlpPolicy',
    #     env=eval_env,
    #     verbose=1,
    #     device='cpu',
    #     tensorboard_log='./runs/',
    # )

    # agent = TD3(
    #     policy='MlpPolicy',
    #     env=eval_env,
    #     verbose=1,
    #     batch_size=1024,
    #     buffer_size=100000,
    #     learning_starts=20000,
    #     tensorboard_log='./runs/',
    # )

    # agent = SAC(
    #     policy='MlpPolicy',
    #     env=eval_env,
    #     verbose=1,
    #     train_freq=128,
    #     batch_size=512,
    #     buffer_size=1000000,
    #     learning_starts=200000,
    #     tensorboard_log='./runs/',
    #     device='cpu',
    # )

    agent = DDPG(
        policy='MlpPolicy',
        env=eval_env,
        verbose=1,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.005,
        train_freq=256,
        learning_starts=25e3,
        tensorboard_log='./runs/',
    )

    agent.learn(total_timesteps=2000000)
    agent.save('runs')
