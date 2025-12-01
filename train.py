import gymnasium as gym
from stable_baselines3 import DQN
import sumo_rl

# Create the SUMO-RL environment
env = gym.make('sumo-rl-v0',
               net_file='quirino_avenue.net.xml',
               route_file='traffic_routes.rou.xml',
               use_gui=False,  # Set to True to watch the simulation
               num_seconds=86400,
               delta_time=15,
               yellow_time=5,
               reward_fn='queue',
               additional_sumo_cmd="--additional-files vehicle_types.add.xml") # Total seconds per episode

# Instantiate the PPO agent from Stable Baselines3
model = DQN('MlpPolicy', 
            env, 
            verbose=1,
            exploration_fraction=0.2,
            exploration_final_eps=0.1,
            gamma=0.999,
            learning_rate=0.0001,
            policy_kwargs={ 'net_arch':[128,128,128,128] },
            tensorboard_log="./tensorboard_logs/")

# Train the agent
model.learn(total_timesteps=250000)
model_path = "./model.zip"
model.save(model_path)

env.close()
