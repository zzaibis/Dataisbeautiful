import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Reading the Datasets
train = pd.read_csv("train_loan.csv")

test = pd.read_csv("test_loan.csv")

pd.options.display.max_columns = train.shape[0]


# Imputing missing values for Train Set
# From here on values in train dataframe will be changed

Gender_mode = train['Gender'].mode()[0]
train['Gender'].fillna(Gender_mode, inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train["Credit_History"].mode()[0], inplace=True)
train['Dependents'].replace('3+', 3,inplace=True)
train = train.drop('Loan_ID', 1)
# print(train.isnull().sum())

# For Test set
Gender_mode = test['Gender'].mode()[0]
test['Gender'].fillna(Gender_mode, inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['Credit_History'].fillna(test["Credit_History"].mode()[0], inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
test = test.drop('Loan_ID', 1)
# print(test.isnull().sum())

def loan_pred(dtrain, dtest, predictors, outcome, model_):
    model_.fit(dtrain[predictors], dtrain[outcome])
    predictors = model_.predict(dtrain[predictors])
    accuracy = accuracy_score(predictors, dtrain[outcome])
    print(accuracy)

predictor_var = ['Credit_History']
outcome_var = ['Loan_Status']
model_var = LogisticRegression()
# loan_pred(train, test, predictor_var, outcome_var, model_var )


def loan_prediction_fn(train_set, test_set,model):
    X = train_set.drop('Loan_Status', 1)
    y = train_set.Loan_Status

    X = pd.get_dummies(X)
    test_set = pd.get_dummies(test_set)
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
    # pred_test = model.predict(test_set)
    # print(pred_test)

# Feature Engineering on Train and test data
train_fe = train
test_fe = test
train_fe['TotalIncome'] = train_fe['ApplicantIncome'] + train_fe['CoapplicantIncome']
test_fe['TotalIncome'] = test_fe['ApplicantIncome'] + test_fe['CoapplicantIncome']

train_fe['TotalIncome_log'] = np.log(train_fe['TotalIncome'])
test_fe['TotalIncome_log'] = np.log(test_fe['TotalIncome'])

train_fe['EMI'] = train_fe['LoanAmount'] / train_fe['Loan_Amount_Term']
test_fe['EMI'] = test_fe['LoanAmount'] / test_fe['Loan_Amount_Term']

train_fe['Balance_Income'] = train_fe['TotalIncome'] - (train_fe['EMI']*1000)
test_fe['Balance_Income'] = test_fe['TotalIncome'] - (test_fe['EMI']*1000)

train_fe = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test_fe = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

# loan_prediction_fn(train, test, LogisticRegression())

loan_prediction_fn(train_fe, test_fe, LogisticRegression())

# loan_prediction_fn(train, test, DecisionTreeClassifier())

# loan_prediction_fn(train_fe, test_fe, DecisionTreeClassifier())

# loan_prediction_fn(train_fe, test_fe, RandomForestClassifier())

