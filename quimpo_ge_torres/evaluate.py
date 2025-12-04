import gymnasium as gym
import sumo_rl
from stable_baselines3 import DQN
import numpy as np
import os
import sys

# ==========================================
# 1. DISABLE LIBSUMO FOR GUI
# ==========================================
# We must remove this env var, otherwise sumo-rl tries to load the C++ library
# which doesn't support the GUI window.
if 'LIBSUMO_AS_TRACI' in os.environ:
    print("⚠️ Disabling LIBSUMO for GUI mode...")
    del os.environ['LIBSUMO_AS_TRACI']
# ==========================================
# 2. DEFINE THE WRAPPER
# ==========================================
# We need the wrapper here so the code calculates the reward
# exactly how the agent expects it (even if we are just watching).
class FairnessRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ts_id = self.env.unwrapped.ts_ids[0]
        ts = self.env.unwrapped.traffic_signals[ts_id]

        # 2. NumPy Math Optimization
        waits = np.array(ts.get_accumulated_waiting_time_per_lane(), dtype=np.float32)
        squared_wait = np.sum(np.square(waits))
        raw_reward = -1.0 * (squared_wait / 5000.0)
        reward = np.clip(raw_reward, -20.0, 0.0)

        return obs, reward, terminated, truncated, info

import numpy as np
from sumo_rl import ObservationFunction
from gymnasium import spaces

class WaitTimeObservationFunction(ObservationFunction):
    """
    Custom Observation: Returns [Queue_Length_Per_Lane, Wait_Time_Per_Lane]
    This allows the agent to SEE how long cars have been waiting.
    """
    def __init__(self, ts):
        super().__init__(ts)

    def __call__(self):
        # 1. Get Queue (Number of stopped cars)
        queues = self.ts.get_lanes_queue() # Returns list of floats
        
        # 2. Get Wait Times (Seconds)
        # normalize by dividing by 1000 roughly to keep numbers small-ish
        waits = [w / 100.0 for w in self.ts.get_accumulated_waiting_time_per_lane()]
        
        # 3. Combine them into one vector
        observation = np.array(queues + waits, dtype=np.float32)
        return observation

    def observation_space(self):
        # We have 2 metrics per lane. 
        # Size = Number of Lanes * 2
        num_features = len(self.ts.lanes) * 2
        return spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(num_features,), 
            dtype=np.float32
        )

# ==========================================
# 3. SETUP ENVIRONMENT
# ==========================================

env = FairnessRewardWrapper(gym.make('sumo-rl-v0',
                       net_file='quimpo-GE_torres.net.xml',
                       route_file='quimpo_traffic_route.rou.xml',
                       use_gui=True,  # <--- WE WANT TO SEE IT
                       num_seconds=86400, # You can simulate a full day now if you want
                       delta_time=15,
                       yellow_time=5,
                       additional_sumo_cmd=f"--additional-files vehicle_types.add.xml"
                       ))

# ==========================================
# 4. RUN EVALUATION
# ==========================================
model_path = "./model.zip"

# Check if model exists
if not os.path.exists(model_path):
    print("Error: model.zip not found! Did you finish training?")
    sys.exit(1)

model = DQN.load(model_path)

print("Starting Evaluation...")


obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # deterministic=True means "Pick the BEST action", no randomness
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated


print(f"Episode Finished. Total Accumulated Fairness Reward: {total_reward}")
env.close()

