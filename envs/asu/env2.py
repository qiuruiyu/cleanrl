from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy import io 
import gymnasium as gym 
from gymnasium import spaces
from gymnasium.wrappers import NormalizeObservation, TimeAwareObservation, NormalizeReward
from gymnasium.utils import seeding
from stable_baselines3 import PPO, TD3, SAC, DQN
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
import torch
import copy
import sys 
import os 
sys.path.append('./')


class ASUEnv(gym.Env):
    def __init__(self,
                 env_config: dict
                 ) -> None:
        """
        System initialization, 7x7 system 
        """
        self.Tsim = env_config['Tsim'] if 'Tsim' in env_config.keys() else 2500
        self.obs_dict = env_config['obs_dict'] if 'obs_dict' in env_config.keys() else False
        self.nu = 4
        self.ny = 7
        self.back = self.Tsim
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
        # self.goal = np.array([
        #     98485, 
        #     18940,
        #     39772, 
        #     1.434+0.1490, 
        #     99.7648-0.0305, 
        #     0.317+0.0260, 
        #     9.57]).reshape(-1, 1)
        self.goal = np.array([
            99000, 
            19000,
            39976, 
            1.57, 
            99.7332, 
            0.3508, 
            9.3461]).reshape(-1, 1)
        
        self.target = self.goal / self.y0
        
        # weight parameters 
        self.Q = np.ones((1, self.ny))
        self.R = np.ones((1, self.nu))

        # step parameters 
        self.S = io.loadmat('./envs/asu/step_4.mat')['Step']
        self.S = self.S[:self.Tsim, :, :]
        self.N = self.S.shape[0]
        self.M = np.block([
            [np.zeros((self.N - 1, 1)), np.eye(self.N - 1)],
            [np.zeros((1, self.N - 1)), 1]
        ])

        '''
        =================== action space =================
        | du1 | du2  | du3  | du4  | du5 |  du6  |  du7  | 
        |  0  |   0  |  0   |  0   |  0  |  0    |  0    |
        |  6  | 4e-3 | 1e-2 | 2e-3 | 3e3 |  600  |  1200 |
        ==================================================
        '''
        # self.action_low = np.array([0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
        # self.action_high = np.array([6, 4e-3, 1e-2, 2e-3, 3e3, 600, 1200]).astype(np.float32)
        self.action_high = np.array([6, 4e-3, 1e-2, 2e-3]).astype(np.float32)
        self.action_high *= 2
        self.action_low = -self.action_high


        self.action_space = spaces.Box(
            # self.action_low, self.action_high,
            -1, 1,
            shape=(self.nu,)
        )
        # self.num_divide = np.array([7, 5, 6, 5, 6, 6, 7])
        # self.num_divide = np.array([7, 7, 7, 7])
        # self.action_space = spaces.MultiDiscrete(self.num_divide)

        '''
        ================= observation space ================ 
        '''
        if self.obs_dict:
            self.observation_space = spaces.Dict(
                spaces={
                    # "goal": spaces.Box(0, np.inf, (self.ny, self.back), dtype=np.float64),
                    "y": spaces.Box(-10, 10, (self.ny, self.back), dtype=np.float32),
                    "error": spaces.Box(-np.inf, np.inf, (self.ny, self.back), dtype=np.float32),
                    "du": spaces.Box(-1, 1, (self.nu, self.back), dtype=np.float32),
                    "u": spaces.Box(-10, 10, (self.nu, self.back), dtype=np.float64),
                } 
            )
        else: 
            # self.obs_low = np.array([-np.inf, -1]).astype(np.float32)
            # self.obs_high = np.array([np.inf, 1]).astype(np.float32)
            self.observation_space = spaces.Box(
                # self.obs_low, self.obs_high,
                -np.inf, np.inf,
                shape=(2*self.ny+2*self.nu, self.back)
            )

        self.state = self.assign_init_state() 

    def assign_init_state(self):
        # goal = np.repeat(self.goal, self.back, axis=1)
        # y = np.repeat(np.zeros((7, 1)), self.back, axis=1)
        # error = np.repeat(np.ones((7, 1)), self.back, axis=1)
        # error = np.repeat(self.target-1, self.back, axis=1)
        # du = np.zeros((self.nu, self.back))
        # u = np.repeat(np.zeros((self.nu, 1)), self.back, axis=1)

        y = np.repeat(self.y0, self.back, axis=1)
        error = np.repeat(self.goal-self.y0, self.back, axis=1)
        du = np.zeros((self.nu, self.back))
        u = np.repeat(np.zeros((self.nu, 1)), self.back, axis=1)


        if self.obs_dict:
            init_state = {
                # "goal": goal,
                "y": y,
                "error": error,
                "du": du,
                "u": u,
            }
        else:
            init_state = np.vstack(
                (
                    y,
                    error,
                    du,
                    u,
                )
            )
        return init_state

    def get_obs_info(self, obs):
        if self.obs_dict:
            obs_info = [obs[key] for key in obs.keys()]
            return tuple(obs_info)
        else:
            y_ = obs[:self.ny, :]
            error_ = obs[self.ny:2*self.ny, :]
            du_ = obs[-2*self.nu:-self.nu, :]
            u_ = obs[-self.nu:, :]
            return tuple([y_, error_, du_, u_]) 
    
    def update_state(self, info:tuple):
        '''
        tuple format like (current_error, action)
        '''
        y_, error_, du_, u_ = self.get_obs_info(self.state)
        current_y, current_error, current_du, current_u = copy.deepcopy(info)
        # current_u /= self.u0

        '''
        hstack current state value
        '''
        y_ = np.hstack((
            # y_[:, 1:], current_y - 1
            y_[:, 1:], current_y,
        ))

        error_ = np.hstack((
            error_[:, 1:], current_error
            # error_[:, 1:], current_error + error_[:, -1].reshape(-1, 1)  # integral error 
        ))
        
        du_ = np.hstack((
            du_[:, 1:], current_du
        ))        
        
        u_ = np.hstack((
            # u_[:, 1:], current_u - 1
            u_[:, 1:], current_u
        ))
        
        '''
        vtstack the time frame for state value 
        '''
        if self.obs_dict:
            self.state = {
                "y": y_,
                "error": error_,
                "du": du_,
                "u": u_,
            }
        else:
            self.state = np.vstack((
                y_, error_, du_, u_,
            ))
    
    def rescale_action(self, action):
        action = action.squeeze() # change to vector 
        # res = self.action_low + (self.action_high - self.action_low) * ((action + 1) / 2)
        res = (action + 1) / 2 * self.action_high
        # res = self.action_low.reshape(-1, 1) + action * ((self.action_high - self.action_low).reshape(-1, 1) / (self.num_divide-1).reshape(-1, 1))
        return res
    
    def calculation_output(self):
        # y = y.reshape(-1, 1)

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
        delta = np.vstack(
            (
                (self.usim[1:self.num_step+1, 3] - self.usim[:self.num_step, 3]).reshape(1, -1),
                (self.ysim[1:self.num_step+1, 0] - self.ysim[:self.num_step, 0]).reshape(1, -1),
                (self.ysim[1:self.num_step+1, 1] - self.ysim[:self.num_step, 1]).reshape(1, -1),
                (self.ysim[1:self.num_step+1, 2] - self.ysim[:self.num_step, 2]).reshape(1, -1),
            )
        )

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
        fig, axs = plt.subplots(7, 1, figsize=(6, 9))
        for i in range(self.ny):
            axs[i].plot(t, self.ysim[:-1, i], 'b-')
            axs[i].plot([0, t[-1]], [self.goal[i], self.goal[i]], 'r--')
        plt.show()

    def step(self, action):
        """
        :param action: 
        :return: tuple(observation, reward, terminated, truncated, info)
        """
        action = action.reshape(-1, 1)
        # action = np.clip(action, np.zeros((self.nu, 1)), self.action_high.reshape(-1, 1))
        self.num_step += 1
        self.du = self.rescale_action(action).reshape(-1, 1)
        # self.du = action
        self.u += self.du
        self.dusim[self.num_step, :] = self.du.squeeze() 
        self.usim[self.num_step, :] = self.u.squeeze() 

        '''
        Optimized NumPy operation, faster x1.25
        '''
        # self.yk = self.M.dot(self.yk)
        # self.yk += np.matmul(self.S, self.du).squeeze()
        # tmp_y = self.yk[0, :].reshape(-1, 1)
        # tmp_y *= 10
        # self.ysim[self.num_step, :] = tmp_y.squeeze() 
        self.calculation_output()

        # current_y = self.ysim[self.num_step, :].reshape(-1, 1) / self.y0
        current_y = self.ysim[self.num_step, :].reshape(-1, 1)
        # current_error = self.target - current_y
        current_error = self.goal - current_y
        # current_error = self.ysim[self.num_step, :].reshape(-1, 1) - self.goal
        self.update_state((current_y, current_error, action, self.u))

        '''
        Reward part & 
        Terminated & Truncated judgement part 
        '''
        reward = 0 
        terminated = False
        truncated = False

        if self.num_step == self.Tsim:
            # self.render_plot()
            truncated = True
        # reward -= np.sum(current_error**2)
        reward -= np.sum(current_error / (self.goal - self.y0))
        # reward -= np.sum(self.Q.dot(current_error**2))
                        #   + self.R.dot(action**2))
        # reward -= np.sum(self.Q.dot(current_error[:-1, :]**2))
                        #   + self.R.dot(action**2))

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
        
        # step yk 
        self.yk = np.repeat(self.y0.T, self.N, axis=0)  # N x yk

        self.state = self.assign_init_state()
        # self.state = self.observation_space.sample() 

        return self.state, {} 


def make_asu_env(rank=0, seed=0):
    # def _init():
    env = ASUEnv(
        {
            'Tsim': 800,
            'obs_dict': False
        }
    )
    env.reset(seed=seed+rank)
    return env
    # set_random_seed(seed)
    # return _init