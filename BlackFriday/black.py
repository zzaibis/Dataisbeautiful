import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# train = pd.read_csv(StringIO('train.csv'), sep='\s+')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

pd.options.display.max_columns = 999

# print(train.head())
# print(train.shape)
# print(test.shape)
#
# # print(train.describe())
#
# # print(train.info())
#
# #Check for duplicates
#
# IDsUnique = len(set(train.User_ID))
# print(IDsUnique)
#
# IdsTotal = len(train)
#
# print(IdsTotal)
#
# IdsDup = IdsTotal - IDsUnique
#
# print(IdsDup)
#
# print('There are ' + str(IdsDup) + ' Duplicate IDs for ' + str(IdsTotal) + ' Total IDs, meaning only ' + str(IDsUnique) + ' of them are unique')

# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(12,7))
# sns.distplot(train['Purchase'], bins=25)
# plt.xlabel('Purchases')
# plt.ylabel('No of Buyers')
# plt.title('Purchase amount Distribution')
# # plt.show()

# Numeric Predictors assesment
numeric_features = train.select_dtypes(include=[np.number])

# Univariate Analysis

# #Occupation
# #Marital_Status
# #Product_Category_1
# #Product_Category_2
# #Product_Category_3
#
# def countplott(obj):
#     return sns.countplot(train[obj])
# print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
# # for i in numeric_features:
# #     if i not in ['User_ID', 'Purchase']:
# #         countplott(i)
# #         print(train[i].value_counts())
# #         plt.show()


# # Correlation between Numeric features and Target variable
#
# # category_features = train.select_dtypes(include=['object'])
# # print(category_features.dtypes)
# #
# # print(train.isnull().sum()/train.shape[0]*100)
# #
# # print(train['Occupation'].unique())
# #
# # sns.countplot(train['Marital_Status'])
# # plt.show()
#
# corr = numeric_features.corr()
#
# print(corr['Purchase'].sort_values(ascending=False))
#
# # Correlation Matrix
# f, ax = plt.subplots(figsize=(20, 9))
# sns.heatmap(corr,  annot=True);
# # plt.show()
#
# # Categorical Predictors
#
categorical_features = train.select_dtypes(include=object)
#
# for i in categorical_features:
#     if i not in ['Product_ID']:
#         sns.countplot(train[i])
#         # plt.show()
#
# # Bivariate Analysis
#
# # Numeric Variables # Pivot tables of target and predictor variables
#
# for i in numeric_features:
#     if i not in ['User_ID', 'Purchase']:
#         pivot_table = train.pivot_table (index=i, values='Purchase', aggfunc=np.mean)
#         print(pivot_table)
#         pivot_table.plot(kind= 'bar')
#         plt.xlabel(i)
#         plt.ylabel('Purchase')
#         plt.title('Average Purchase among categories')
#         plt.xticks(rotation= 1)
#         # plt.show()
#
#
# # Categorical Variables
#
# for i in categorical_features:
#     if i not in ['Product_ID']:
#         pivot_table = train.pivot_table (index=i, values='Purchase', aggfunc=np.mean)
#         print(pivot_table)
#         pivot_table.plot(kind= 'bar')
#         plt.xlabel(i)
#         plt.ylabel('Purchase')
#         plt.title('Average Purchase among categories')
#         plt.xticks(rotation= 1)
#         # plt.show()

# DATA PREPROCESSING

# Combining the dataset into one dataframe for easy cleaning and feature engineering
train['source'] = 'train'
test['source'] = 'test'

black_friday = pd.concat([train, test], ignore_index=True, sort=False)
print(train.shape, test.shape, black_friday.shape)

# Looking at the missing values for imputation

print(black_friday.isnull().sum())

# Imputing values to missing values


black_friday['Product_Category_2'].fillna(0, inplace=True)
black_friday['Product_Category_3'].fillna(0, inplace=True)








