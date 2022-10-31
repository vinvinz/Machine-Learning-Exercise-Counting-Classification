import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
import numpy as np
import string

csv_val = pd.read_csv("test_dataset.csv")

df = csv_val[['label', 'x0', 'y0']]

fig, ax = plt.subplots(figsize=(5,5))
ax = sns.scatterplot(x='x0',y='y0',hue = 'label',data = df,legend='full',
                     palette = {1:'red',2:'orange',3:'blue',4:'green'})
ax.legend(loc='lower left')
plt.show()
# print(df.head())