import scipy.stats as stats
import researchpy as rp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
df = pd.read_pickle('dataframe.pkl')
df=df[df['totalasset']>30]
result=df.pivot_table(index=['buy_tresh','sell_tresh','buysellratio'], aggfunc='size')
result=result[result==4]
plt.hist([i[0] for i in result.keys()])