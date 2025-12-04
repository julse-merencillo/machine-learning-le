import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import sumo_rl
import numpy as np

# 1. Force Libsumo for speed
os.environ['LIBSUMO_AS_TRACI'] = '1'

class FairnessRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ts_id = self.env.unwrapped.ts_ids[0]
        ts = self.env.unwrapped.traffic_signals[ts_id]

        # 2. NumPy Math Optimization
        waits = np.array(ts.get_accumulated_waiting_time_per_lane(), dtype=np.float32)
        max_wait_time = np.max(waits) if len(waits) > 0 else 0.0

        if max_wait_time > 200:
            reward = -50.0
        else:
            squared_wait = np.sum(np.square(waits))
            raw_reward = -1.0 * (squared_wait / 5000.0)
            reward = np.clip(raw_reward, -10.0, 0.0)

        return obs, reward, terminated, truncated, info

def make_env():
    # Helper function to create the env for parallelization
    env = gym.make('sumo-rl-v0',
                   net_file='quimpo-GE_torres.net.xml',
                   route_file='quimpo_traffic_route.rou.xml',
                   use_gui=False,
                   num_seconds=14400, # REDUCED: 24h (86400) is too long for one episode!
                   delta_time=15,
                   yellow_time=5,
                   additional_sumo_cmd="--additional-files vehicle_types.add.xml")
    env = FairnessRewardWrapper(env)
    return env

if __name__ == "__main__":
    # 3. Create 4 environments running in parallel on different CPU cores
    # This collects 4x the data in the same amount of time
    vec_env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)

    model = DQN('MlpPolicy',
                vec_env,
                verbose=1,
                buffer_size=50000,
                learning_starts=1000,
                target_update_interval=500, # Update target net less frequently for stability
                max_grad_norm = 0.5,
                exploration_fraction=0.3,
                exploration_final_eps=0.02,
                gamma=0.99,
                learning_rate=0.00005,
                policy_kwargs={'net_arch': [256, 256]},
                tensorboard_log="./tensorboard_logs/")

    model.learn(total_timesteps=250000)
    model.save("model")
    print("Training Complete")
