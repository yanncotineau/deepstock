import pandas as pd
import numpy as np
from StockEnv0 import StockEnv0

def Random_games(env, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)


df = pd.read_csv('./data/pricedata.csv')
df = df.sort_values('Date')

lookback_window_size = 10
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

train_env = StockEnv0(train_df, lookback_window_size=lookback_window_size)
test_env = StockEnv0(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, train_episodes = 10, training_batch_size=500)
