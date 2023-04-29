from env1 import Env1
import pandas as pd
from stable_baselines3 import PPO

df_path = '../data/pricedata.csv'
df = pd.read_csv(df_path).sort_values('Date')

lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

TIMESTEPS = 1e6
test_episodes = 10

# setting up env
train_env = Env1(train_df)
test_env = Env1(test_df)
train_env.reset()

# training model
model = PPO("MlpPolicy", train_env, verbose=1)
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
for i in range(test_episodes):
    average_net_worth += test(test_env, False) / test_episodes

print(f"average_net_worth : {average_net_worth}")
        
