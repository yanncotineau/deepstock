from env1 import Env1
import pandas as pd

df_path = '../data/pricedata.csv'
df = pd.read_csv(df_path).sort_values('Date')

lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

episodes = 1000

env = Env1(test_df)

average_net_worth = 0
for episode in range(episodes):
    print(f"episode {episode} :")
    state = env.reset()

    while True:
        env.render()

        action = env.action_space.sample()

        state, reward, done , _ = env.step(action)

        if done:
            average_net_worth += env.net_worth / episodes
            print("\tnet_worth:", env.net_worth)
            break
    print("------------------------------")
print("\naverage_net_worth:", average_net_worth)