import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_canada = pd.read_excel('Canada.xlsx', sheet_name='Canada by Citizenship', skiprows=range(20), dtype=object)

df_canada = pd.read_excel('Canada.xlsx', sheet_name='Canada by Citizenship', skiprows=range(20), index_col= 'OdName', dtype=object)

df_japan = df_canada.ix['Japan', 8:43]

print(df_japan)

df_japan.plot(kind='box')

# df_canada.ix['Haiti', 8:43].plot(kind='pie')
df_canada['Total'] = df_canada.iloc[:, 8:43].sum(axis=1)
df_continents = df_canada.groupby(by='AreaName', axis=0).sum()

# df_continents['Total'].plot(kind='box')


plt.show()

pd.options.display.max_columns = df_canada.shape[0]

# print(df_canada.columns)

df_canada['Total'] = df_canada.iloc[:, 8:43].sum(axis=1)

df_top5 = df_canada.sort_values(['Total'], ascending=False, axis=0).head()