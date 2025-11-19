import sumo_rl
import gymnasium as gym
from stable_baselines3 import DQN

print("\n--- Running the trained model ---")

# 1. Load the environment with the GUI enabled
eval_env = gym.make('sumo-rl-v0',
                    net_file='single_intersection.net.xml',
                    route_file='single_intersection.rou.xml',
                    use_gui=True, # Set to True to watch the agent
                    num_seconds=10000)

model_path = "./model.zip"

# 2. Load the trained model
saved_model = DQN.load(model_path, env=eval_env)

# 3. Run the simulation loop
obs, info = eval_env.reset()
done = False
while not done:
    # 4. Get the best action from the model (deterministic=True)
    #    deterministic=True ensures the agent doesn't explore and picks the best-known action.
    action, _states = saved_model.predict(obs, deterministic=True)
    print(f'Action: {action}') 
    # 5. Apply the action to the environment
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

eval_env.close()
print("Evaluation finished.")
