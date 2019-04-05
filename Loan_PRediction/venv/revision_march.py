import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

loan_train = pd.read_csv('train_loan.csv')
loan_train['source'] = 'train'
loan_test = pd.read_csv('test_loan.csv')
loan_test['source'] = 'test'
pd.options.display.max_columns = loan_train.shape[0]

print(loan_train.shape)
print(loan_test.shape)

loan_data = pd.concat([loan_train, loan_test], sort=False)

print(loan_data.shape)

# Missing Values
print(loan_train.isnull().sum())

print(loan_train['Gender'].value_counts(dropna=False))

gender_freq = loan_train['Gender'].value_counts()

pd.crosstab(loan_train['Education'], loan_train['Gender']).plot(kind = 'bar', stacked = True)

plt.show()

