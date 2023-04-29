from env1 import Env1
import pandas as pd
from stable_baselines3 import PPO

# constants
timesteps = 1e5
lookback_window_size = 50
test_episodes = 100

# getting and cleaning data
df_path = '../data/pricedata.csv'
df = pd.read_csv(df_path).sort_values('Date')
train_df = df[:-500-lookback_window_size]
test_df = df[-500-lookback_window_size:] # 30 days

# setting up train env
train_env = Env1(train_df)
train_env.reset()

# training model
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

def test(env, deterministic):
    state = env.reset(0)
    while True:
        env.render()
    
        action, _ = model.predict(state, deterministic=deterministic)
    
        state, reward, done , _ = env.step(action)
    
        if done:
            print("\tnet_worth:", env.net_worth)
            return env.net_worth

# setting up test env
test_env = Env1(test_df)

# testing model
state = test_env.reset(0)

# testing model
average_net_worth = 0
average_net_worth += test(test_env, True) / test_episodes
for i in range(test_episodes - 1):
    average_net_worth += test(test_env, False) / test_episodes

print(f"average_net_worth : {average_net_worth}")
        
