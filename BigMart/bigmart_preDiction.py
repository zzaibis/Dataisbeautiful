import pandas as pd
import numpy as np
from scipy.stats  import mode
import matplotlib.pyplot as plt
import seaborn as sns

Train = pd.read_csv("Train.csv")
print(Train.shape)
Test = pd.read_csv("Test.csv")
print(Test.shape)
bigmart = pd.concat([Train, Test], ignore_index=True, sort=False)
pd.options.display.max_columns = bigmart.shape[0]


print(bigmart.shape)

# print(bigmart.describe())

# Index(['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
#        'Item_Type', 'Item_MRP', 'Outlet_Identifier',
#        'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
#        'Outlet_Type', 'Item_Outlet_Sales'],
#       dtype='object')

print(bigmart.columns)
print(bigmart.isnull().sum())


# Imputing Missing values of Item_Weight & Outlet_Size

# Item_Weight

item_avg_weight = bigmart.pivot_table(values='Item_Weight', index='Item_Identifier')

print(len(item_avg_weight))

miss_bool = bigmart['Item_Weight'].isnull()

bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Weight'] = bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Identifier'].apply(lambda x:  item_avg_weight.loc[x])

print(bigmart.isnull().sum())

# Outlet_Size

outlet_size_mode = bigmart.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x: x.mode().iat[0]))
print(outlet_size_mode)

bigmart.loc[bigmart['Outlet_Size'].isnull(), 'Outlet_Size'] = bigmart.loc[bigmart['Outlet_Size'].isnull(), 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print(bigmart.isnull().sum())

