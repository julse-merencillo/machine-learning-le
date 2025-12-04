# check_network.py
import gymnasium as gym
import sumo_rl
import os

os.environ['LIBSUMO_AS_TRACI'] = '1'

env = gym.make('sumo-rl-v0',
               net_file='quimpo-GE_torres.net.xml',
               route_file='quimpo_traffic_route.rou.xml',
               use_gui=True,  # <--- Watch this with your eyes
               num_seconds=3600,
               delta_time=20,
               yellow_time=4,
               additional_sumo_cmd="--additional-files vehicle_types.add.xml")

obs, info = env.reset()
done = False
ts_id = env.unwrapped.ts_ids[0]
ts = env.unwrapped.traffic_signals[ts_id]

print("Starting Round-Robin Test...")
while not done:
    # Force cycle through phases: 0 -> 1 -> 2 -> 0 ...
    current_phase = ts.green_phase()
    next_action = (current_phase + 1) % ts.num_green_phases
    
    obs, reward, terminated, truncated, info = env.step(next_action)
    done = terminated or truncated

env.close()