import numpy as np
import matplotlib.pyplot as plt
import random
from src.env_v2 import MinesweeperEnvV2 as MinesweeperSmartEnv
from src.agent import QAgent

BOARD_SIZE = 5
N_MINES = 3
EPISODES = 20000

env = MinesweeperSmartEnv(size=BOARD_SIZE, n_mines=N_MINES)
agent = QAgent(actions=[0, 1], alpha=0.1, gamma=0.9, epsilon=1.0)
history_rewards = []
win_count = 0

print(f"--- Iniciando Entrenamiento Modelo 2.0 ({EPISODES} episodios) ---")

for episode in range(EPISODES):
    env.reset()
    total_reward = 0
    
    # Primer paso seguro
    safe_starts = list(zip(*np.where(env.grid != -1)))
    sr, sc = random.choice(safe_starts)
    env.view[sr, sc] = env.grid[sr, sc]
    env.revealed_count += 1

    while not env.done:
        candidates = env.get_valid_cells()
        if not candidates: break
        
        target_cell = random.choice(candidates)
        state = env.get_context_state(target_cell[0], target_cell[1])
        
        action = agent.choose_action(state)
        reward, done = env.step(action, target_cell)
        next_state = env.get_context_state(target_cell[0], target_cell[1])
        
        agent.update(state, action, reward, next_state)
        total_reward += reward

    agent.decay_epsilon()
    history_rewards.append(total_reward)
    if env.revealed_count == (BOARD_SIZE**2 - N_MINES):
        win_count += 1

    if (episode + 1) % 2000 == 0:
        print(f"Episodio {episode+1}: Recompensa media: {np.mean(history_rewards[-100:]):.2f} | Victorias acumuladas: {win_count}")

moving_avg = np.convolve(history_rewards, np.ones(100)/100, mode='valid')
plt.figure(figsize=(10,5))
plt.plot(moving_avg, label='Modelo 2.0 (Smart Context)')
plt.title(f"Evolución Modelo 2.0 - Victorias Totales: {win_count}")
plt.xlabel("Episodios")
plt.ylabel("Recompensa Media")
plt.legend()
plt.show()