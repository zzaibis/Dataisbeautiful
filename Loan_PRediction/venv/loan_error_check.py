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


# Imputing missing values
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train["Credit_History"].mode()[0], inplace=True)
train['Dependents'].replace('3+', 3, inplace=True)
table = train.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]
train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

# Feature Engineering
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['TotalIncome_log'] = np.log(train['TotalIncome'])
train['EMI'] = train['LoanAmount']/ train['Loan_Amount_Term']
train['payback'] = train['LoanAmount']/ train['TotalIncome']
train = train.drop('Loan_ID', 1)

print(train.isnull().sum())

# print(train.info())

categoricals = [x for x in train.dtypes.index if train.dtypes[x] == 'object']
continuous = [ x for x in train.dtypes.index if train.dtypes[x] == 'int64']
cat = [x for x in train.dtypes.index if train.dtypes[x] == 'float64']

print(categoricals)

print(train.dtypes)


def loan_prediction_fn(train_set, test_set,model):
    X = train_set.drop('Loan_Status', 1)
    y = train_set.Loan_Status

    X = pd.get_dummies(X)
    test_set = pd.get_dummies(test_set)
    all_score = []
    print(X.head())

    i = 1
    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        xtrain, xtest = X.loc[train_index], X.loc[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        model.fit(xtrain, ytrain)
        pred_test = model.predict(xtest)
        score = accuracy_score(ytest, pred_test)
        # print('accuracy_score', score)
        all_score.append(score)
        i += 1
    print(np.mean(all_score))
    # pred_test = model.predict(test_set)
    # print(pred_test)

loan_prediction_fn(train, test, LogisticRegression())