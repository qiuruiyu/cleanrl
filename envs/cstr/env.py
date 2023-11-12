import numpy as np 
from scipy import io 
import scipy.linalg
from scipy.integrate import odeint
import gymnasium as gym 
from gym.utils import seeding
import torch
from copy import copy
from typing import Tuple, Union

class CSTR(gym.Env):
    def __init__(self, env_config:dict) -> None:
        self.Tsim = env_config['Tsim'] if 'Tsim' in env_config.keys() else 100 
        self.Q = env_config['Q'] if 'Q' in env_config.keys() else None 
        self.R = env_config['R'] if 'R' in env_config.keys() else None 

        self.nu = 1 
        self.ny = 1
        self.num_steps = 0 
        