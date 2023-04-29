import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np
import imageio

class Renderer:
    def __init__(self, render_range):

        
        # Get a list of all the files in the folder
        self.folder_path = "./temp_frames"

        file_list = os.listdir(self.folder_path)
        
        # Loop through the list and delete each file
        for file_name in file_list:
            file_path = os.path.join(self.folder_path, file_name)
            os.remove(file_path)
        
        self.render_range = render_range
        self.volume = deque(maxlen=render_range)
        self.net_worth = deque(maxlen=render_range)
        self.render_data = deque(maxlen=render_range)
        
        self.frame_counter = 0
        
        plt.style.use('ggplot')
        plt.close('all')
        self.fig = plt.figure(figsize=(10,5))
        
        # Axe supérieur
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
        
        # Axe inférieur
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
        
        # Axe pour le net_worth
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')
        
    def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):
        
        self.volume.append(Volume)
        self.net_worth.append(net_worth)
        
        
        Date = mpl_dates.date2num([pd.to_datetime(Date)])[0]
        self.render_data.append([Date, Open, High, Low, Close])
        
        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/24, colorup='green', colordown='red', alpha=0.8)

        
        Date_Render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(Date_Render_range, self.volume, 0)

        
        self.ax3.clear()
        self.ax3.plot(Date_Render_range, self.net_worth, color="blue")
        
        
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in Date_Render_range:
                if trade['type'] == 'buy':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="v")

        
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        self.fig.tight_layout()

        frame_counter_string = "{:0>{}}".format(self.frame_counter, 4)
        
        plt.savefig(f"temp_frames/" + frame_counter_string + ".png")
        self.frame_counter += 1
        
    def merge_frames(self):
        file_list = sorted([os.path.join(self.folder_path, file_name) for file_name in os.listdir(self.folder_path) if file_name.endswith('.png')])

        # Create the GIF using imageio
        with imageio.get_writer('gifs/animation.gif', mode='I', duration=0.25) as writer:
            for file_path in file_list:
                image = imageio.imread(file_path)
                writer.append_data(image)