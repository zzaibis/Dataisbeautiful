import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
# Importing Predictive models' modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



df = pd.read_csv("train_loan.csv")
df_test = pd.read_csv("test_loan.csv")
pd.options.display.max_columns = df.shape[0]

y = df.Loan_Status
X = df.drop('Loan_Status', axis=1)

# print(X.shape)
# print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state= 0)
# print(X_train)
# print(X_test.head())
# print(y_train)
# print(y_test.head())



# Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
#       dtype='object')

#
# df['split'] = np.random.randn(df.shape[0], 1)
# msk = np.random.rand(len(df)) <= 0.7
# train = df[msk]
# test = df[~msk]
pd.options.mode.chained_assignment = None  # default='warn'

# print(train.shape)
# print(test.shape)

# Replace missing values
#

table = X_train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
X_train["Self_Employed"].fillna("No", inplace=True)
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

#
X_train['LoanAmount'].fillna(X_train[X_train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
X_train["TotalIncome"] = X_train["ApplicantIncome"] + X_train["CoapplicantIncome"]
X_train["LoanAmount_log"] = np.log(X_train["LoanAmount"])
X_train["TotalIncome_log"] = np.log(X_train["TotalIncome"])
X_train["TotalIncome_log"].hist(bins=25)
X_train["Paybackrate"] = (X_train["LoanAmount"] /X_train["TotalIncome"])*100
X_train['Gender'].fillna(X_train['Gender'].mode()[0], inplace=True)
X_train['Married'].fillna(X_train['Married'].mode()[0], inplace=True)
X_train['Dependents'].fillna(X_train['Dependents'].mode()[0], inplace=True)
X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mode()[0], inplace=True)
X_train['Credit_History'].fillna(X_train['Credit_History'].mode()[0], inplace=True)

# print(X_train.apply(lambda x: sum(x.isnull()), axis=0))



# sklearn requires all inputs to be numeric, we should convert all our categorical variables into numeric by encoding the categories
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area',  'TotalIncome', 'LoanAmount_log', 'TotalIncome_log', 'Paybackrate']
le = LabelEncoder()
for i in var_mod:
    X_train[i] = le.fit_transform(X_train[i])
X_train.dtypes


table = X_test.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
X_test["Self_Employed"].fillna("No", inplace=True)
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

#
X_test['LoanAmount'].fillna(X_test[X_test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
X_test["TotalIncome"] = X_test["ApplicantIncome"] + X_test["CoapplicantIncome"]
X_test["LoanAmount_log"] = np.log(X_test["LoanAmount"])
X_test["TotalIncome_log"] = np.log(X_test["TotalIncome"])
X_test["TotalIncome_log"].hist(bins=25)
X_test["Paybackrate"] = (X_test["LoanAmount"] /X_test["TotalIncome"])*100
X_test['Gender'].fillna(X_test['Gender'].mode()[0], inplace=True)
X_test['Married'].fillna(X_test['Married'].mode()[0], inplace=True)
X_test['Dependents'].fillna(X_test['Dependents'].mode()[0], inplace=True)
X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mode()[0], inplace=True)
X_test['Credit_History'].fillna(X_test['Credit_History'].mode()[0], inplace=True)

# print(X_test.apply(lambda x: sum(x.isnull()), axis=0))


X_train.set_index("Loan_ID", inplace=True)
X_test.set_index("Loan_ID", inplace=True)


# sklearn requires all inputs to be numeric, we should convert all our categorical variables into numeric by encoding the categories
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area',  'TotalIncome', 'LoanAmount_log', 'TotalIncome_log', 'Paybackrate']
le = LabelEncoder()
for i in var_mod:
    X_test[i] = le.fit_transform(X_test[i])
X_test.dtypes

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))



# The predictive model function
def classification_model(model, data,data_test, predictors, outcome):
    # Fit the model
    model.fit(data[predictors], outcome)

    # Make predictions on X_training set
    predictions = model.predict(data[predictors])

    # Print Accuracy
    accuracy = metrics.accuracy_score(predictions, outcome)
    print(accuracy)

    model.score(data_test[predictors],)





    # Fit the model again so that it can be refered outside the function:
    # model.fit(data[predictors], outcome)

# Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status',
#        'TotalIncome', 'LoanAmount_log', 'TotalIncome_log', 'Paybackrate'],

outcome_var = y_train
model = LogisticRegression()

predictor_var = ['Credit_History']
test_data = X_test
classification_model(model, X_train, y_train, predictor_var, outcome_var)
