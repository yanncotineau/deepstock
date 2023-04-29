from StockEnv1 import StockEnv1
import pandas as pd
from stable_baselines3 import PPO

episodes = 50
training_batch_size=500

df = pd.read_csv('./data/pricedata.csv')
df = df.sort_values('Date')

lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

env = StockEnv1(train_df, lookback_window_size=lookback_window_size)
env.reset()

'''average_net_worth = 0
for episode in range(episodes):
    state = env.reset(env_steps_size = training_batch_size)

    while True:
        env.render()

        action = env.action_space.sample()

        state, reward, done , _ = env.step(action)

        if env.current_step == env.end_step:
            average_net_worth += env.net_worth
            print("net_worth:", env.net_worth)
            break

print("average_net_worth:", average_net_worth/episodes)'''

model = PPO("MlpPolicy", env, verbose=1)
TIMESTEPS = 10000
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)