from env1 import Env1
import pandas as pd
from stable_baselines3.common.env_checker import check_env

df_path = '../data/pricedata.csv'
df = pd.read_csv(df_path).sort_values('Date')

env = Env1(df)
check_env(env)