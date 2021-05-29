import pandas as pd
import datetime as dt
import yfinance as yf

#get dates
end = dt.datetime.today()
start = dt.datetime(end.year - 10, end.month, end.day)

#get dataframe
tickers = ['AAPL', 'AMZN', 'GOOG', 'FB', 'MSFT']
df = yf.download(tickers, start, end)['Adj Close']

df.to_csv("prices.csv")