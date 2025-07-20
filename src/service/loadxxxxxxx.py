import pandas as pd

def download(symbol, start_date, end_date):
    # df = yf.download(symbol, start=start_date, end=end_date)
    df = pd.read_csv('data.csv')
    print(df.head())
