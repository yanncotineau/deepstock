## Libraries

import os
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from polygon import RESTClient

## Constants and setup

data_directory = "../data"
intraday_minute_length = 390
os.chdir(data_directory)

## Utils functions

def get(ticker = "AAPL", p = 0.7, save_on_dl = True):
    # This functions returns some train and test dataframes for intraday stock trading.
    # Each dataframe is a compilation of random 390-step wide days
    
    if os.path.isfile(ticker + ".csv"): # if data has already been downloaded for that ticker
        df = pd.read_csv(ticker + ".csv", index_col=None) # just return it
    else:
        df = download(ticker = ticker) # else download it using polygon.io stock API
        if save_on_dl:
            df.to_csv(ticker + ".csv", index=None)
    return df


def download(ticker = "AAPL"):
    # Constants
    API_KEY = "OjN7BD9lkIoCDcvzuRXWMw_93iugp6QH"
    LIMIT = 50000
    multiplier = 1
    timespan = "minute"
    from_ = "2000-01-01"
    to = datetime.now().strftime('%Y-%m-%d')    
    
    client = RESTClient(api_key=API_KEY)
    
    data = client.get_aggs(ticker, multiplier, timespan, from_, to, limit=LIMIT)
    df = pd.DataFrame(data)
    
    # get data
    last_minute = 0
    while data[-1].timestamp > last_minute:
        time.sleep(12) # polygon.io API only allows for 5 calls per minute
        last_minute = data[-1].timestamp # Last minute in response
        last_minute_date = datetime.fromtimestamp(last_minute/1000).strftime('%Y-%m-%d')
        print(f"last minute date : {last_minute_date}")
        try:
            data = client.get_aggs(ticker, multiplier, timespan,
                                      last_minute_date, to,
                                      limit=LIMIT)
            
            new_bars = pd.DataFrame(data)
            df = df.append(new_bars[new_bars.timestamp > last_minute])
        except:
            break
        
    # clean data
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["Date"] = df["Datetime"].dt.date
    
    df["Time"] = df["Datetime"].dt.time
    
    df = df.drop(columns=["Datetime", "timestamp", "transactions", "otc", "vwap"])
    df.columns = ["Open", "High", "Low", "Close", "Volume", "Date", "Time"]
    
    
    # Supprimer les journées non complètes
    value_counts = df['Date'].value_counts()
    
    # Suppression des lignes dont le nombre d'occurrences est inférieur à 390
    df = df[df['Date'].map(value_counts) >= intraday_minute_length]
    
    df.reset_index()
    
    return df

df = get("MSFT")
#df = get("AAPL")
#df = get("TSLA")