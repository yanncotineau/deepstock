## Libraries

import numpy as np
from numpy import float64 as FLOAT
import random
from collections import deque
import gym
import gym.spaces as spaces
import pandas as pd

from ta.momentum import rsi
from ta.trend import sma_indicator
from ta.trend import macd

from renderer import Renderer


## Constants

DEFAULT_INITIAL_BALANCE = 1000
DEFAULT_LOOKBACK_WINDOW_SIZE = 50
DEFAULT_EPISODE_MAX_STEPS = 500
BROKERAGE_FEE = 0.01
STOP_LOSS = 0.9

RSI_RANGE = 15
SMA_RANGE = 20
MACD_RANGE = 20

RENDER_RANGE = 100

## Environment class

class Env2(gym.Env):

    def __init__(self, df, initial_balance = DEFAULT_INITIAL_BALANCE, lookback_window_size = DEFAULT_LOOKBACK_WINDOW_SIZE, visual = False):
        
        self.df = df.dropna().reset_index()

        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.visual = visual

        self.action_space = spaces.Discrete(3)
        self.state_size = (self.lookback_window_size, 13)
        self.observation_space = spaces.Box(
            low = 0, high = np.inf, shape=(self.state_size), dtype=FLOAT
        )

        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)

    def reset(self, episode_max_steps = DEFAULT_EPISODE_MAX_STEPS):
    
        if self.visual:
            self.renderer = Renderer(render_range=RENDER_RANGE)
        self.trades = deque(maxlen=RENDER_RANGE)
    
        self.balance = FLOAT(self.initial_balance)
        self.net_worth = FLOAT(self.initial_balance)
        self.prev_net_worth = self.initial_balance
        self.stock_bought = FLOAT(0)
        self.stock_held = FLOAT(0)
        self.stock_sold = FLOAT(0)
        
        self.punish_value = FLOAT(0)

        if episode_max_steps > 0: # selecting a random episode_max_steps-wide window in the dataset to train
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - episode_max_steps)
            self.end_step = self.start_step + episode_max_steps - 1
        else: # testing on entire dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step
        
        # Initializing market_history
        self.normalize_prices(self.current_step)
        
        # Initializing orders_history
        for i in reversed(range(self.lookback_window_size)):
            self.orders_history.append([self.balance, self.net_worth, self.stock_bought, self.stock_held, self.stock_sold, FLOAT(0), FLOAT(0), FLOAT(0)])

        state = np.concatenate((self.market_history, self.orders_history), axis = 1)
        return state

    def step(self, action):
        self.stock_bought = FLOAT(0)
        self.stock_sold = FLOAT(0)

        current_price = self.df.loc[self.current_step, 'Close']
        current_date = self.df.loc[self.current_step, 'Date']
        current_high = self.df.loc[self.current_step, 'High']
        current_low = self.df.loc[self.current_step, 'Low']

        if action == 0: # Hold
            pass

        elif action == 1 and self.stock_held == 0:
            
            # including brokerage fees when buying
            total_current_price = current_price * (1 + BROKERAGE_FEE)
            
            self.stock_bought = self.balance / total_current_price 
            self.balance -= self.stock_bought * total_current_price
            self.stock_held = self.stock_bought
            
            self.trades.append({'Date' : current_date, 'High' : current_high, 'Low' : current_low, 'total': self.stock_bought, 'type': "buy"})

        elif action == 2 and self.stock_held > 0:
            
            # including brokerage fees when selling
            total_current_price = current_price * (1 - BROKERAGE_FEE)
            
            self.stock_sold = self.stock_held
            self.balance += self.stock_sold * total_current_price
            self.stock_held = FLOAT(0)
            
            self.trades.append({'Date' : current_date, 'High' : current_high, 'Low' : current_low, 'total': self.stock_sold, 'type': "sell"})

        # new net worths
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price


        # reward
        if action == 0:
            self.punish_value += self.net_worth * 0.001
            reward = - self.punish_value
        else:
            self.punish_value = FLOAT(0)
            reward = self.net_worth - self.prev_net_worth
        
        
        

        if self.net_worth <= self.initial_balance * STOP_LOSS: # condition d'arrêt d'échec
            done = True
        elif self.current_step == self.end_step: # condition d'arrêt temps limite
            done = True
        else: # continue to next step
            done = False
            self.current_step += 1

        # getting new observation
        self.normalize_prices(self.current_step)
        
        # ajout des indicateurs à partir des prix normalisés
        normalized_closes = pd.Series([x[3] for x in self.market_history])
        
        current_RSI = FLOAT(rsi(normalized_closes[-RSI_RANGE:], RSI_RANGE).iloc[-1]/100)
        current_SMA = FLOAT(sma_indicator(normalized_closes[-SMA_RANGE:], SMA_RANGE).iloc[-1])
        current_MACD = FLOAT(macd(normalized_closes[-MACD_RANGE:], MACD_RANGE).iloc[-1])
        
        self.orders_history.append([self.balance, self.net_worth, self.stock_bought, self.stock_held, self.stock_sold, current_RSI, current_SMA, current_MACD])
        
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)

        info = {}
        
        if done and self.visual:
            self.renderer.merge_frames()
        
        return obs, reward, done, info

    def render(self):
        if self.visual:
            
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']
            
            self.renderer.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)
            
    # Normaliser les lookback_window_size derniers prix selon une étape actuelle donnée
    def normalize_prices(self, current_step):
        
        
        
        # Getting prices data from dataframe
        for i in reversed(range(self.lookback_window_size)):
            ith_previous_step = current_step - i
            
            self.market_history.append([self.df.loc[ith_previous_step, 'Open'],
                                        self.df.loc[ith_previous_step, 'High'],
                                        self.df.loc[ith_previous_step, 'Low'],
                                        self.df.loc[ith_previous_step, 'Close'],
                                        self.df.loc[ith_previous_step, 'Volume']
                 
                                        ])

        # Normalizing the first 4 columns (O, H, L, C)
        arr = np.array(self.market_history)
        
        x_min = np.min(arr[:, :4]).astype(FLOAT)
        x_max = np.max(arr[:, :4]).astype(FLOAT)
        
        arr[:, :4] = (arr[:, :4].astype(FLOAT) - x_min) / (x_max - x_min)
        
        self.market_history = deque(arr, maxlen=self.lookback_window_size)
