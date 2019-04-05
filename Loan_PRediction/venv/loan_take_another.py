import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Reading the Datasets
train = pd.read_csv("train_loan.csv")

test = pd.read_csv("test_loan.csv")

pd.options.display.max_columns = train.shape[0]

# Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
#       dtype='object')

# Imputing values on train set

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
# train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train["Credit_History"].mode()[0], inplace=True)
train['Dependents'].replace('3+', 3, inplace=True)
train = train.drop('Loan_ID', 1)
table = train.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

train['TotalIncome_log'] = np.log(train['TotalIncome'])

# train = train.drop('TotalIncome', 1)
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]
train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
train['EMI'] = train['LoanAmount']/ train['Loan_Amount_Term']

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['payback'] = train['LoanAmount']/ train['TotalIncome']




# le = LabelEncoder()
#
# categorical_vars = [x for x in train.dtypes.index if train.dtypes[x] == 'object']
#
# print(categorical_vars)
# for obj in categorical_vars:
#     train[obj] = le.fit_transform(train[obj])




def loan_prediction_fn(train_set, model):
    X = train_set.drop('Loan_Status', 1)
    y = train_set.Loan_Status

    X = pd.get_dummies(X)
    all_score = []

    i = 1
    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        # print('\n{} of kfold {}'.format(i, kf.n_splits))
        xtrain, xtest = X.loc[train_index], X.loc[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        model.fit(xtrain, ytrain)
        pred_test = model.predict(xtest)
        score = accuracy_score(ytest, pred_test)
        # print('accuracy_score', score)
        all_score.append(score)
        i += 1
    print(np.mean(all_score))


# loan_prediction_fn(train, LogisticRegression())






# predictor_var = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']

predictor_var = ['Credit_History', 'TotalIncome_log', 'EMI', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term',  'payback' ]

predictor_var = ['Credit_History']
outcome_var = 'Loan_Status'

# predictor_var = ['Credit_History']


# def loan_predict( model, dtrain, predictors, outcome):
#
#     X = dtrain[predictors]
#     y = dtrain[outcome]
#
#     # X = pd.get_dummies(X)
#
#     all_score = []
#     kf = StratifiedKFold(n_splits=5, random_state=1)
#     for train_index, test_index in kf.split(X, y):
#         xtrain, xtest = X.loc[train_index], X.loc[test_index]
#         ytrain, ytest = y[train_index], y[test_index]
#
#         model.fit(xtrain, ytrain)
#         pred_test = model.predict (xtest)
#         score = accuracy_score(ytest, pred_test)
#         # mse = mean_squared_error(ytest, pred_test)
#         # print(mse)
#         all_score.append(score)
#     print(np.mean(all_score))

# loan_predict(LogisticRegression(), df_train, predictor_var, outcome_var)


df_train = train

def another_model(model_name , dtrain, predictors, outcome):
    X = dtrain[predictors]
    y = dtrain[outcome]
    print(X.shape)
    print(y.shape)

    X = pd.get_dummies(X)

    skfold = StratifiedKFold(n_splits=5, random_state=1)
    model_name.fit(X, y)
    pred_vals = model_name.predict(X)
    cv_score = cross_val_score(model_name, X, y, cv=skfold)
    print(np.mean(cv_score))

another_model(LogisticRegression(), df_train, predictor_var, outcome_var)

