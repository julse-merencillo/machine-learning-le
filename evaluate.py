# evaluate_agent.py
import gymnasium as gym
import sumo_rl
from stable_baselines3 import DQN
import os

# Load the environment using the RL config (the one WITHOUT the .ttl.xml)
env = gym.make('sumo-rl-v0',
               net_file='quirino_avenue.net.xml',
               route_file='traffic_routes.rou.xml',
               use_gui=True,  # Set to True to watch your agent work!
               num_seconds=86400,
               delta_time=15,
               yellow_time=5,
               out_csv_name='rl_metrics.csv',
               additional_sumo_cmd=f"--additional-files vehicle_types.add.xml --seed 42 --tripinfo-output rl_trip_info.xml" # USE THE SAME SEED!
               )

# Load the trained model
model_path = "./model.zip"
model = DQN.load(model_path)

print("Starting RL Agent evaluation...")
obs, info = env.reset()
done = False
while not done:
    # Use deterministic=True for evaluation to get the best action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
