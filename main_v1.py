import numpy as np
import matplotlib.pyplot as plt
import random
from env_v1 import MinesweeperEnvV1 as MinesweeperSmartEnv
from src.agent import QAgent

env = MinesweeperSmartEnv(size=6, n_mines=6)
agent = QAgent(actions=[0, 1], alpha=0.1, gamma=0.9, epsilon=1.0)
EPISODES = 5000
rewards_history = []
win_count = 0

print(f"Iniciando entrenamiento ({EPISODES} partidas)...")

for episode in range(EPISODES):
    env.reset()
    total_reward = 0
    
    safe_cells = list(zip(*np.where(env.grid != -1)))
    if safe_cells:
        sr, sc = random.choice(safe_cells)
        env.view[sr, sc] = env.grid[sr, sc]
        env.revealed_count += 1

    pasos = 0
    while not env.done and pasos < 100:
        pasos += 1
        candidates = []
        for r in range(env.size):
            for c in range(env.size):
                if env.view[r, c] == -1:
                    ctx = env.get_context_vector(r, c)
                    if any(v >= 0 for v in ctx):
                        candidates.append((r, c, ctx))

        if not candidates:
            hidden = list(zip(*np.where(env.view == -1)))
            if not hidden: break
            r, c = random.choice(hidden)
            state = env.get_context_vector(r, c)
        else:
            r, c, state = random.choice(candidates)

        action = agent.choose_action(state)
        reward, done = env.step(action, r, c)
        next_state = env.get_context_vector(r, c)
        agent.update(state, action, reward, next_state)
        total_reward += reward

    agent.decay_epsilon()
    rewards_history.append(total_reward)
    if env.revealed_count == (env.size**2 - env.n_mines):
        win_count += 1

    if (episode + 1) % 500 == 0:
        media = np.mean(rewards_history[-100:])
        print(f"Episodio {episode+1} | Recompensa Media: {media:.1f} | Epsilon: {agent.epsilon:.2f}")

print(f"Entrenamiento finalizado. Victorias: {win_count}")

window = 100
avg_rewards = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
plt.plot(avg_rewards)
plt.title("Aprendizaje del Agente")
plt.show()