import copy
import sys
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.wrappers import(NormalizeObservation, NormalizeReward, TimeAwareObservation)
from scipy import io

class ASUEnvG3(gym.Env):
    def __init__(self,
                 env_config: dict
                 ) -> None:
        """
        System initialzation, part --> 3x3 system 
        """
        self.Tsim = env_config['Tsim'] if 'Tsim' in env_config.keys() else 2500
        # self.obs_dict = env_config['obs_dict'] if 'obs_dict' in env_config.keys() else False
        self.nu = 3 
        self.ny = 3 
        self.back = env_config['back'] if 'back' in env_config.keys() else 100
        self.total_reward = [] 
        self.u0 = np.array([
            93285,  # 空气调节
            56.466, # 氧气调节
            47.4,   # 氮气调节
        ]).reshape(-1, 1)
        self.y0 = np.array([
            93285,    # 空气量
            17940,    # 氧气量
            37672,    # 氮气量
        ]).reshape(-1, 1)

        self.goal = np.array([
            99000, 
            19000,
            39976, 
        ]).reshape(-1, 1)

        # init projective useful variables 
        self.e0_real = self.goal - self.y0 
        self.e0 = (self.goal - self.y0) / self.goal
        self.target = np.zeros((self.ny, 1))

        self.S = io.loadmat('./envs/asu/S3.mat')['S']
        if self.Tsim < self.S.shape[0]:
            self.S = self.S[:self.Tsim, :, :]
        self.N = self.S.shape[0]
        self.M = np.block([
            [np.zeros((self.N - 1, 1)), np.eye(self.N - 1)],
            [np.zeros((1, self.N - 1)), 1]
        ])

        # ACTION SPACE 
        self.action_high = np.array([
            6, 4e-3, 1e-2
        ]).astype(np.float32)
        self.action_low = -self.action_high

        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            shape=(self.nu,),
        )

        # OBSERVATION SPACE
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ny+self.nu, self.back),
            dtype=np.float32,
        )

        self.state = self.assign_init_state() 

    def assign_init_state(self):
        error = np.repeat(self.e0, self.back, axis=1)
        du = np.zeros((self.nu, self.back)) 
        u = np.repeat(self.u0, self.back, axis=1)
        return np.vstack((error, du))
    
    def step(self, action):
        """step to the next state"""
        action = action.reshape(-1, 1)
        self.num_step += 1
        # rescale action 
        self.du = action * self.action_high
        self.u += self.du 
        
        # update state
        self.yk = self.M.dot(self.yk)
        self.yk += np.matmul(self.S, self.du).squeeze() 
        tmp_y = self.yk[0, :].reshape(-1, 1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """function for reset of the env"""
        if seed is not None:
            super().reset(seed=seed)  

        self.num_step = 0 
        self.total_reward.append(0)
        # reset monitoring params 
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

        # step used 
        self.yk = np.repeat(self.y0.T, self.N, axis=0)  # N x yk 
        
        self.state = self.assign_init_state()

        return self.state, {} 
        
    



if __name__ == "__main__":
    env = ASUEnvG3(
        {
            'Tsim': 3000,
        }
    )
    print(env.S)