from env2 import Env2
import pandas as pd
from stable_baselines3.common.env_checker import check_env

print("1")

df_path = '../data/pricedata.csv'
df = pd.read_csv(df_path).sort_values('Date')

print("2")

env = Env2(df)

print("3")

env.normalize_prices(150)



print("4")

# check_env(env)