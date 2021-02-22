import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sns

plt.style.use('dark_background')

df = pd.read_csv('/Users/spusegao/Downloads/AirPassengers.csv')
print(df.dtypes)

df['Month'] = pd.to_datetime(df['Month'])
print(df.dtypes)
