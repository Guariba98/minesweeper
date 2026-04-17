import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class MinesweeperEnvPPO(gym.Env):
    """Entorno de Buscaminas compatible con Stable Baselines 3 (Gymnasium)"""
    def __init__(self, size=6, n_mines=6):
        super(MinesweeperEnvPPO, self).__init__()
        self.size = size
        self.n_mines = n_mines
        
        self.action_space = spaces.Discrete(self.size * self.size)

        self.observation_space = spaces.Box(
            low=-2, high=8, shape=(1, self.size, self.size), dtype=np.float32
        )
        
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.size, self.size), dtype=int)
        mines_idx = random.sample(range(self.size**2), self.n_mines)
        for idx in mines_idx:
            r, c = divmod(idx, self.size)
            self.grid[r, c] = -1
            
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == -1: continue
                count = sum(1 for dr, dc in self.directions if 0 <= r+dr < self.size and 0 <= c+dc < self.size and self.grid[r+dr, c+dc] == -1)
                self.grid[r, c] = count
        
        self.view = np.full((self.size, self.size), -1, dtype=np.float32)
        self.revealed_count = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.view], dtype=np.float32)

    def step(self, action):
        r, c = divmod(action, self.size)
        reward = 0
        terminated = False
        truncated = False
        
        if self.view[r, c] != -1:
            reward = -2
        elif self.grid[r, c] == -1:
            reward = -10
            terminated = True
        else:
            self.view[r, c] = self.grid[r, c]
            self.revealed_count += 1
            reward = 1
            if self.revealed_count == (self.size**2 - self.n_mines):
                reward = 20
                terminated = True
        
        return self._get_obs(), reward, terminated, truncated, {}