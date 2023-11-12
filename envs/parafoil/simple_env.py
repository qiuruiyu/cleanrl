import gymnasium as gym 
import numpy as np 
from typing import Dict, Tuple, Any, Union, Optional
import matplotlib.pyplot as plt 
from gymnasium import spaces
import copy 


class Parafoil(gym.Env):
    def __init__(
            self,
            start_point: Tuple = (-600,-600, 900),
            umax: float = 0.14,
            gamma0: float = 0.75*np.pi,
            vs: float = 15,
            vz: float = -4.6,
            dt: float = 0.1,
            nstack: int = 1,
    ) -> None:
        """
        Create an env demo of Parafoil
        :param start_point: starting point of parafoil
        :param umax: max bounding of input
        :param gamma0: initial rad value of parafoil
        :param vs: initial velocity on x-y coordinate
        :param vz: initial velocity on z axis
        :param dt: simulation time step
        :param nstack: number of frames to stack
        """
        self.nu = 1
        self.ny = 3
        self.nobs = 7

        self.start_point = np.array(start_point).reshape(self.ny, 1)
        self.umax = umax  # omega, omega = \dot \gamma 
        self.gamma0 = gamma0
        self.vs = vs
        self.vz = vz
        self.dt = dt
        self.goal = np.zeros((self.ny, 1))

        self.nstack = nstack
        self.num_steps = 0
        self.traj = [] 

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nobs, self.nstack),
        )
        """
        ======= obs =======
        x:
        y: 
        z: 
        vs:
        vz:
        gamma:
        u: 
        ===================
        """

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.nu,),
        )

    def _init_obs(self):
        self.current_state = np.vstack((
            self.start_point,
            self.vs,
            self.vz,
            self.gamma0,
            0
        )).reshape(-1, 1)
        self.state = self._scale_obs(self.current_state)
        self.state = np.repeat(self.state, self.nstack, axis=1)

    def _scale_obs(self, obs: np.ndarray, inverse=False):
        # assert obs.shape == (self.nobs, self.nstack), "obs shape should be (nobs, nstack)"
        obs = obs.copy()
        if not inverse:
            obs[:3, -1] /= 600
        else:
            obs[:3, -1] *= 600
        return obs

    def _update_state(self, du):
        dx = self.state[3, -1] * np.cos(self.state[5, -1]) * self.dt
        dy = self.state[3, -1] * np.sin(self.state[5, -1]) * self.dt
        dz = self.state[4, -1] * self.dt
        dgamma = du * self.dt
        self.state[:, :-1] = self.state[:, 1:]
        self.state[:, -1] = np.vstack((
            self.state[0, -1] + dx / 600,
            self.state[1, -1] + dy / 600,
            self.state[2, -1] + dz / 600,
            self.vs,
            self.vz,
            self.state[5, -1] + dgamma,
            du,
        )).reshape(self.nobs,)
        self.traj.append(self.state[:3, -1])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        :param action: action to take
        :return: next observation, reward, terminated, truncated, info
        """
        self.num_steps += 1
        du = action * self.umax
        self._update_state(du)

        reward = 0
        terminated = False
        truncated = False

        '''
        reward calculation 
        '''

        # reward for moving towards goal
        reward -= (
            4 * np.sum(self.state[:2, -1]**2) / (np.abs(self.state[2, -1]) * 600) + 
            2 * np.sum(du**2)
        ) 

        # reward for moving towards np.pi angle 
        reward -= 2 * np.abs(self.state[5, -1] - np.pi) / (np.abs(self.state[2, -1]) * 600)

        if self.state[2, -1] < 1e-6:
            print(f'parafoil landed at: ({self.state[0, -1]}, {self.state[1, -1]}, {self.state[2, -1]})')
            terminated = True
            
            # reward for landing at goal (terminal reward)
            if np.abs(self.state[0, -1]) < 0.5 and np.abs(self.state[1, -1]) < 0.5:
                reward += (np.exp(-(self.state[:2, -1]**2)) * 100 - 65).sum()

        if self.num_steps > 10000:
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            super().reset(seed=seed)

        self.start_point = np.random.uniform(-600, 600, size=(3, 1))
        self.start_point[-1] = 900
        self.num_steps = 0
        self.traj = [] 
        self._init_obs()

        return self.state, {} 


def make_parafoil_env():
    return Parafoil(dt=1, nstack=1)


if __name__ == "__main__":
    env = Parafoil(dt=1)

    from stable_baselines3 import PPO, SAC, HerReplayBuffer
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

    agent = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        # tensorboard_log='./logs/',
    )

    agent.learn(total_timesteps=1000000, progress_bar=True)

    # agent.save('./models/agent')

    # traj = [] 
    # agent = PPO.load('./models/agent.zip')

    # next_obs, _ = env.reset() 
    # reward = 0 
    # while True: 
    #     traj.append(np.array(next_obs[:3, -1]))
    #     # print(next_obs[:3, -1])
    #     print(next_obs[-1, -1])
    #     action, _ = agent.predict(next_obs)
    #     next_obs, r, done, _, _ = env.step(action)
    #     reward += r 
    #     if done: 
    #         print('env done')
    #         break

    # traj = np.array(traj) * 900
    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(3, 1, figsize=(3, 6))
    # for i in range(3):
    #     axs[i].plot(traj[:, i])

    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # ax.scatter(traj[0, 0], traj[0, 1], s=20, c='r')
    # ax.scatter(traj[-1, 0], traj[-1, 1], s=20, c='g')
    # ax.scatter(traj[:, 0], traj[:, 1], s=1)
    # plt.grid()
    # plt.show() 

