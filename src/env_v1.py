import numpy as np
import random

class MinesweeperEnvV1:
    def __init__(self, size=6, n_mines=6):
        self.size = size
        self.n_mines = n_mines
        self.directions = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        posibles = range(self.size * self.size)
        minas = random.sample(posibles, self.n_mines)
        for m in minas:
            r, c = divmod(m, self.size)
            self.grid[r, c] = -1

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == -1: continue
                count = 0
                for dr, dc in self.directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.grid[nr, nc] == -1:
                            count += 1
                self.grid[r, c] = count

        self.view = np.full((self.size, self.size), -1)
        self.done = False
        self.revealed_count = 0
        return self

    def get_context_vector(self, r, c):
        neighborhood = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighborhood.append(self.view[nr, nc])
            else:
                neighborhood.append(-3) # "Fuera del tablero"
        return tuple(neighborhood)

    def step(self, action, r, c):
        reward = 0
        if action == 0: # Revelar
            if self.grid[r, c] == -1:
                reward = -100
                self.done = True
                self.view[r, c] = -1
            else:
                if self.view[r, c] == -1:
                    self.view[r, c] = self.grid[r, c]
                    self.revealed_count += 1
                    reward = 5
                    if self.grid[r, c] == 0: reward += 2
                else:
                    reward = -5
        elif action == 1: # Bandera
            if self.grid[r, c] == -1:
                if self.view[r, c] != -2:
                    self.view[r, c] = -2
                    reward = 20
            else:
                reward = -50

        safe_cells = (self.size**2) - self.n_mines
        if self.revealed_count == safe_cells:
            reward += 100
            self.done = True

        return reward, self.done