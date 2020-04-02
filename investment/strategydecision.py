import scipy.stats as stats
import researchpy as rp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_pickle('dataframe.pkl')
df=df[df['totalasset']>30]
result=df.pivot_table(index=['buysellratio','buy_tresh','sell_tresh'], aggfunc='size')
result=result[result==result.max()]
plt.hist([i[0] for i in result.keys()])