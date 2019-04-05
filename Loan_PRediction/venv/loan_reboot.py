import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('train_loan.csv')

warnings.filterwarnings('ignore')

test = pd.read_csv('test_loan.csv')

pd.options.display.max_columns = train.shape[0]

train['source'] = 'train'
test['source'] = 'test'


loan = pd.concat([train, test], ignore_index=True)

print(loan.keys())

print(loan['Credit_History'].value_counts())
print(loan['Self_Employed'].value_counts(dropna=False))

credit_history = pd.crosstab(loan['Credit_History'], loan['Education'], normalize=True)

loan['TotalIncome'] = loan['ApplicantIncome'] + loan['CoapplicantIncome']

avg_salary = pd.pivot_table(loan, values='TotalIncome', index='Self_Employed', columns=['Education'])

# Education         Graduate  Not Graduate
# Self_Employed
# No             6961.367284   4792.508287
# Yes            8879.787234   6484.400000


loan['group'] = pd.cut(loan.TotalIncome, bins=[0, 5500, 6500, 7500, 8500, 81000 ],labels=['NGNS', 'NGYS', 'YGNS', 'YGYS', 'OTH'])


print(avg_salary)

print(loan.isnull().sum())

# print(loan.group.value_counts())

print(loan.loc[loan['group'] == 'YGYS', 'TotalIncome'])

def fage(x):
    return avg_salary.loc[x['Self_Employed'], x['Education']]
