import pandas as pd
import numpy as np
from scipy.stats  import mode
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.preprocessing import LabelEncoder

train = pd.read_csv("Train.csv")

test = pd.read_csv("Test.csv")

train['source']='train'
test['source']='test'

bigmart = pd.concat([train, test],  ignore_index=True, sort=False)

pd.options.display.max_columns = bigmart.shape[0]
# print(bigmart.describe())
# print(train.shape, test.shape, bigmart.shape)
# print(bigmart.isnull().sum())


# Index(['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
#        'Item_Type', 'Item_MRP', 'Outlet_Identifier',
#        'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
#        'Outlet_Type', 'Item_Outlet_Sales', 'source'],
#       dtype='object')

# print(bigmart.apply(lambda x: list(x.unique())))

def find_unique_values(data, colmn):
    uniq = data[colmn].unique()
    return uniq

# What are the unique values
categorical_columns = [x for x in bigmart.dtypes.index if bigmart.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

# for obj in columns:
#     print("***********************************")
#     print(obj)
#     print(find_unique_values(bigmart, obj))
#     print(len(find_unique_values(bigmart, obj)))
#     print("########################################")

# bigmart_count = bigmart['Outlet_Size'].value_counts()
# print(bigmart_count)


def value_counts_df(data, colmn):
    counts = data[colmn].value_counts()
    return  counts

#
# for obj in categorical_columns:
#     print("***********************************")
#     print(obj)
#     print(value_counts_df(bigmart, obj))
#     print(len(find_unique_values(bigmart, obj)))
#     print("########################################")

# Data Cleaning
# Outlier Detections

continous_columns = [x for x in bigmart.dtypes.index if bigmart.dtypes[x]=='float64']

# print(continous_columns)

# item_avg_weight = bigmart.pivot_table(values='Item_Weight', index='Item_Identifier')
#
# print(len(item_avg_weight))
#
# print(bigmart['Item_Weight'].isnull().sum())
#
# print(len(bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Weight']))
#
# bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Weight'] = bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])

item_avg_group = bigmart.groupby(by ='Item_Identifier')

item_avg_weight_group = item_avg_group['Item_Weight'].agg(np.mean)

bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Weight'] = bigmart.loc[bigmart['Item_Weight'].isnull(), 'Item_Identifier'].apply(lambda x: item_avg_weight_group.loc[x])

# print(bigmart.isnull().sum())

# two ways to find mean categorywise


#
# # outlet_size_mode = bigmart.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:mode(x).mode[0]))
# print(outlet_avg_size)

# outlet_avg_group = bigmart.groupby(by='Outlet_Typ
# outlet_size_mode = bigmart.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:mode[x].mode[0]))
outlet_size_mode = bigmart.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x: x.mode().iat[0]))

# print(outlet_size_mode)

bigmart.loc[bigmart['Outlet_Size'].isnull(), 'Outlet_Size'] = bigmart.loc[bigmart['Outlet_Size'].isnull(), 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

# print(bigmart.isnull().sum())

# Modify Item Visibility, we can see there are several rows with 0 value, Item visibility can't be 0

item_vis = bigmart.pivot_table(values='Item_Visibility', index='Item_Identifier', aggfunc=(lambda x: np.mean(x)))

# print(len(bigmart.loc[bigmart['Item_Visibility'] == 0, 'Item_Identifier']))

bigmart.loc[bigmart['Item_Visibility'] == 0, 'Item_Visibility'] = bigmart.loc[bigmart['Item_Visibility'] == 0, "Item_Identifier"].apply(lambda x: item_vis.loc[x])



# print(len(bigmart.loc[bigmart['Item_Visibility'] == 0, 'Item_Identifier']))
# print(item_vis)

# Feature Engineering
'''
1) See if you can combine Supermarket types in Outlet_Type column
2) Outlet Establishment should be more comparable, the older the outlet the better their relevance
3) Items with Low Fat content gives provides Sales
4) Item_Visibility Mean ratio across stores
5)  
'''


# Creating a new variable consisting the ratio of Item Visibity and the aerage item visibility of a particular Item (Item_Identifier)
bigmart['Item_Visibility_MeanRatio'] = bigmart.apply(lambda x: x['Item_Visibility']/item_vis.loc[x['Item_Identifier']], axis=1)


# Merging Categories of Item_type using Item_Identifier
bigmart['Item_Type_Combined'] = bigmart['Item_Identifier'].apply(lambda x: x[0:2])
bigmart['Item_Type_Combined'] = bigmart['Item_Type_Combined'].replace({
    'FD' : 'FOOD',
    'DR' : 'DRINKS',
    'NC' : 'Non-Consumables'
})


bigmart['Outlet_Years'] = 2013 - bigmart['Outlet_Establishment_Year']


print(bigmart.loc[bigmart['Outlet_Years'].max(), ['Outlet_Establishment_Year', 'Outlet_Size']])


bigmart['Item_Fat_Content'] = bigmart['Item_Fat_Content'].replace({
    'low fat' : 'Low Fat',
    'LF' : 'Low Fat',
    'reg': 'Regular'
})


bigmart.loc[bigmart['Item_Type_Combined'] == 'Non-Consumables', 'Item_Fat_Content'] = 'Non-Edible'

print(bigmart['Item_Fat_Content'].value_counts())


categorical_cols = [x for x in bigmart.dtypes.index if bigmart.dtypes[x]=='object']
# x = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'source', 'Item_Type_Combined']
y = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
continous_cols = [x for x in bigmart.dtypes.index if bigmart.dtypes[x]=='float64']

le = LabelEncoder()

bigmart['Outlet'] = bigmart['Outlet_Identifier']

for obj in y:
    bigmart[obj] = le.fit_transform(bigmart[obj])

print(bigmart)


bigmart = pd.get_dummies(bigmart, columns=y)


# Remove Useless columns and divide test and train data
bigmart.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

bigmart_train = bigmart.loc[bigmart.source == 'train']
bigmart_test = bigmart.loc[bigmart.source == 'test']

bigmart_train.drop(['source'], axis = 1, inplace =True)

bigmart_test.drop(['source', 'Item_Outlet_Sales'], axis = 1, inplace =True)


# bigmart_train.to_csv("bigmart_train.csv", index= False)
# bigmart_test.to_csv("bigmart_test.csv", index= False)






