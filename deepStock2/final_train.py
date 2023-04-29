from env2 import Env2
import pandas as pd
from stable_baselines3 import PPO

ticker = "TSLA"
p = 0.7 # train test split

df_path = f'../data/{ticker}.csv'
df = pd.read_csv(df_path).sort_values('Date')
N = df.shape[0]

lookback_window_size = 50
train_df = df[:-int(N*(1-p))-lookback_window_size]
test_df = df[-int(N*p)-lookback_window_size:]

TIMESTEPS = 1e4
test_episodes = 10

# setting up env
train_env = Env2(train_df)
test_env = Env2(test_df)



# training model
train_env.reset()
model = PPO("MlpPolicy", train_env, verbose=1)

'''model_path = f'./models/{ticker}/model39.zip'
model = PPO.load(model_path)
model.set_env(train_env)'''

for i in range(50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"./models/{ticker}/model{i}")

# testing model
def test(env, deterministic):
    state = env.reset()
    while True:
        env.render()
    
        action, _ = model.predict(state, deterministic=deterministic)
    
        state, reward, done , _ = env.step(action)
    
        if done:
            print("\tnet_worth:", env.net_worth)
            return env.net_worth
        

average_net_worth = 0
for i in range(test_episodes):
    average_net_worth += test(test_env, False) / test_episodes

print(f"average_net_worth : {average_net_worth}")
            
#new_test_env = Env2(test_df, visual=True)
#test(new_test_env, False)
