## Libraries

import numpy as np
from numpy import float64 as FLOAT
import random
from collections import deque
import gym
import gym.spaces as spaces

## Constants

DEFAULT_INITIAL_BALANCE = 1000
DEFAULT_LOOKBACK_WINDOW_SIZE = 50
DEFAULT_EPISODE_MAX_STEPS = 500

## Environment class

class Env1(gym.Env):

    def __init__(self, df, initial_balance = DEFAULT_INITIAL_BALANCE, lookback_window_size = DEFAULT_LOOKBACK_WINDOW_SIZE, step_log = False):
        
        self.df = df.dropna().reset_index()

        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.step_log = step_log

        self.action_space = spaces.Discrete(3)
        self.state_size = (self.lookback_window_size, 10)
        self.observation_space = spaces.Box(
            low = 0, high = np.inf, shape=(self.state_size), dtype=FLOAT
        )

        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)

    def reset(self, episode_max_steps = DEFAULT_EPISODE_MAX_STEPS):
    
        self.balance = FLOAT(self.initial_balance)
        self.net_worth = FLOAT(self.initial_balance)
        self.prev_net_worth = self.initial_balance
        self.stock_bought = FLOAT(0)
        self.stock_held = FLOAT(0)
        self.stock_sold = FLOAT(0)

        if episode_max_steps > 0: # selecting a random episode_max_steps-wide window in the dataset to train
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - episode_max_steps)
            self.end_step = self.start_step + episode_max_steps - 1
        else: # testing on entire dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.stock_bought, self.stock_held, self.stock_sold])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume']
                                        ])
            
            #self.orders_history.append([FLOAT(0), FLOAT(0), FLOAT(0), FLOAT(0), FLOAT(0)])
            #self.market_history.append([FLOAT(0), FLOAT(0), FLOAT(0), FLOAT(0), FLOAT(0)])

        state = np.concatenate((self.market_history, self.orders_history), axis = 1)
        return state

    def step(self, action):
        self.stock_bought = FLOAT(0)
        self.stock_sold = FLOAT(0)

        # CHANGE TO IMPROVE PERFORMANCE
        current_price = self.df.loc[self.current_step, 'Close']

        # GENERATING NEW ORDER BASED ON ACTION
        if action == 0: # Hold
            pass

        # WE BUY A FLOAT AMOUNT OF STOCKS !
        elif action == 1 and self.stock_held == 0:
            self.stock_bought = self.balance / current_price
            self.balance -= self.stock_bought * current_price
            self.stock_held = self.stock_bought

        elif action == 2 and self.stock_held > 0:
            self.stock_sold = self.stock_held
            self.balance += self.stock_sold * current_price
            self.stock_held = FLOAT(0)

        # new net worths
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.stock_bought, self.stock_held, self.stock_sold])

        # reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance/2: # condition d'arrêt d'échec
            done = True
        elif self.current_step == self.end_step: # condition d'arrêt temps limite
            done = True
        else: # continue to next step
            done = False
            self.current_step += 1

        # getting new observation
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)

        info = {}
        return obs, reward, done, info

    def render(self):
        if self.step_log:
            print(f'Step: {self.current_step - self.start_step}, Net Worth: {self.net_worth}')