from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.env_ppo import MinesweeperEnvPPO
import os

BOARD_SIZE = 6
N_MINES = 6
LOG_DIR = "./models/"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

env = MinesweeperEnvPPO(size=BOARD_SIZE, n_mines=N_MINES)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    device="auto"
)


print("Entrenando Agente PPO... Esto puede tardar unos minutos.")
model.learn(total_timesteps=50000)

model.save(f"{LOG_DIR}ppo_minesweeper_v3")
print(f"Modelo guardado en {LOG_DIR}")

obs, _ = env.reset()
print("\n--- Partida de prueba del Agente Entrenado ---")
for _ in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        print("Fin de la partida de prueba.")
        break