import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('train_loan.csv')

test = pd.read_csv('test_loan.csv')

train['source'] = 'train'

test['source'] = 'test'

pd.options.display.max_columns = train.shape[0]
loan = pd.concat([train, test], ignore_index=True, sort=False)

print(loan.shape)
print(train.shape)
print(test.shape)


print(train.isnull().sum())

df_pt = pd.pivot_table(train, values='LoanAmount',index=['Self_Employed', 'Education'], aggfunc='median')
df_pt2 = pd.pivot_table(train, values='LoanAmount',index='Self_Employed',columns='Education', aggfunc='median')
# print(pd.crosstab(columns=train['Credit_History'], values=train['Loan_Status'], index='Education', aggfunc='count'))

train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

train['Self_Employed'].fillna('No', inplace=True)



def fage(x):
    return df_pt2.loc[x['Self_Employed'], x['Education']]

train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Dependents'].replace('3+', 3, inplace=True)
print(train['Dependents'])
print(train.isnull().sum())

train['TotalIncome_log'] = np.log(train['TotalIncome'])

# train['TotalIncome_log'].hist(bins=20)

plt.show()



predictor_var = ['Credit_History']
outcome_var = ['Loan_Status']
model_used = LogisticRegression()

def model_fn(dtrain, dtest, predictors, outcome, model_name):
    model_name.fit(dtrain[predictors], dtrain[outcome])
    predictios = model_name.predict(dtrain[predictors])
    accuracy_score = metrics.accuracy_score(predictios, dtrain[outcome])
    print(accuracy_score)


# model_fn(train, test, predictor_var, outcome_var, model_used )
print(train.keys())
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status', 'TotalIncome', 'TotalIncome_log']
# vars = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'Credit_History', 'Property_Area']

le = LabelEncoder()

for obj in vars:
    train[obj] = le.fit_transform(train[obj])