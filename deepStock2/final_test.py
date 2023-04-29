import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from env2 import Env2
import pandas as pd
from stable_baselines3 import PPO

ticker = "AAPL"
p = 0.7 # train test split
test_episodes = 10

df_path = f'../data/{ticker}.csv'
df = pd.read_csv(df_path).sort_values('Date')
N = df.shape[0]

lookback_window_size = 50
test_df = df[-int(N*p)-lookback_window_size:]

model_path = f'./models/{ticker}/model45.zip'

model = PPO.load(model_path)
test_env = Env2(test_df, visual=True)

def test(env, deterministic):
    state = env.reset()
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