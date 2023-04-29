from env1 import Env1
import pandas as pd
from stable_baselines3 import PPO

df_path = '../data/pricedata.csv'
df = pd.read_csv(df_path).sort_values('Date')
TIMESTEPS = 1e5
test_episodes = 100

# setting up env
env = Env1(df)
env.reset()

# training model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

def test(env, deterministic):
    state = env.reset(0)
    while True:
        env.render()
    
        action, _ = model.predict(state, deterministic=deterministic)
    
        state, reward, done , _ = env.step(action)
    
        if done:
            print("\tnet_worth:", env.net_worth)
            return env.net_worth
        
# testing model
average_net_worth = 0
average_net_worth += test(env, True) / test_episodes
for i in range(test_episodes - 1):
    average_net_worth += test(env, False) / test_episodes

print(f"average_net_worth : {average_net_worth}")
        
