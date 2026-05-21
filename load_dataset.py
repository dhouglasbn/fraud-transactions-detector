import pandas as pd

TRANSACTIONS_FILEPATH = 'transactions_dataset.csv'

def load_dataset():
  try:
    df = pd.read_csv(TRANSACTIONS_FILEPATH)
    return df
  except FileNotFoundError:
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    df.to_csv(TRANSACTIONS_FILEPATH, index=False)
    return df
