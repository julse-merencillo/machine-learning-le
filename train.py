import gymnasium as gym
from stable_baselines3 import DQN
import sumo_rl

# Create the SUMO-RL environment
env = gym.make('sumo-rl-v0',
               net_file='single_intersection.net.xml',
               route_file='single_intersection.rou.xml',
               use_gui=False,  # Set to True to watch the simulation
               num_seconds=100000) # Total seconds per episode

# Instantiate the PPO agent from Stable Baselines3
model = DQN('MlpPolicy', 
            env, 
            verbose=1, 
            tensorboard_log="./tensorboard_logs/")

# Train the agent
model.learn(total_timesteps=10000)
model_path = "./model.zip"
model.save(model_path)

env.close()
