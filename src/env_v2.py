import numpy as np
import random

class MinesweeperEnvV2:
    def __init__(self, size=5, n_mines=3):
        self.size = size
        self.n_mines = n_mines
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        mines_idx = random.sample(range(self.size**2), self.n_mines)
        for idx in mines_idx:
            r, c = divmod(idx, self.size)
            self.grid[r, c] = -1
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == -1: continue
                self.grid[r, c] = sum(1 for dr, dc in self.directions if 0 <= r+dr < self.size and 0 <= c+dc < self.size and self.grid[r+dr, c+dc] == -1)
        self.view = np.full((self.size, self.size), -1)
        self.done = False
        self.revealed_count = 0
        return self

    def get_context_state(self, r, c):
        state = [self.view[r+dr, c+dc] if 0 <= r+dr < self.size and 0 <= c+dc < self.size else -3 for dr, dc in self.directions]
        return tuple(state)

    def get_valid_cells(self):
        hidden = [(r, c) for r in range(self.size) for c in range(self.size) if self.view[r, c] == -1]
        frontier = [cell for cell in hidden if any(self.view[cell[0]+dr, cell[1]+dc] >= 0 for dr, dc in self.directions if 0 <= cell[0]+dr < self.size and 0 <= cell[1]+dc < self.size)]
        return frontier if frontier else hidden

    def step(self, action, cell):
        r, c = cell
        if action == 0: # REVELAR
            if self.grid[r, c] == -1:
                self.done = True
                return -100, True
            self.view[r, c] = self.grid[r, c]
            self.revealed_count += 1
            win = self.revealed_count == (self.size**2 - self.n_mines)
            return (500, True) if win else (10, False)
        elif action == 1: # BANDERA
            self.view[r, c] = -2
            return (50, False) if self.grid[r, c] == -1 else (-20, False)
        return 0, False