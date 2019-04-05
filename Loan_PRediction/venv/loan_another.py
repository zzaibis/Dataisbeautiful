import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train_loan.csv')
test = pd.read_csv('test_loan.csv')

pd.options.display.max_columns = train.shape[0]

# print(train.isnull().sum())

#################################### The Categorical Variables #########################################################

categorical_columns = [x for x in train.dtypes.index if train.dtypes[x] == 'object']
categorical_columns = [x for x in categorical_columns if x not in ['Loan_ID']]

print(categorical_columns)

# Value Counts
def value_counts_cols(dataframe, col):
    return dataframe[col].value_counts(dropna=False)

# for obj in categorical_columns:
#     print(value_counts_cols(train, obj))

temp1 = train['Credit_History'].value_counts(dropna=True, normalize=True)
temp2 = train.pivot_table(values='Loan_Status', index=['Credit_History'],
                          aggfunc=lambda x: x.map({'Y': 1, 'N': 0}).mean())

temp3 = pd.crosstab([train['Credit_History'], train['Gender']], [train['Loan_Status']])

# temp3.plot(kind='bar', stacked=True, color=['red', 'blue'])
# Filling missing values for train set

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Education'].fillna(train['Gender'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Dependents'].replace('3+', 3,inplace=True)
train = train.drop('Loan_ID', 1)

# CATEGORICAL AND CONTINUOUS


pivot1 = train.pivot_table(values='LoanAmount', index=['Education', 'Self_Employed'])
pivot2 = train.pivot_table(values='LoanAmount', index=['Education'], columns='Self_Employed')
# print(pivot1)
# print(pivot2)


table = train.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

# print(table)


def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]


train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

###################################### The Continuous Variables #######################################################

# train['CoapplicantIncome'].hist(bins=50)

train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

# train['TotalIncome'].hist(bins=50)

train['TotalIncome_log'] = np.log(train['TotalIncome'])
train['LoanAmount_log'] = np.log(train['LoanAmount'])


# sns.distplot(train['LoanAmount_log'], rug=True, color='red', norm_hist=True, axlabel='Howdy')


#### Feature Engineering

train['EMI'] = train['LoanAmount']/ train['Loan_Amount_Term']

train['payback_ratio'] = train['EMI']/train['TotalIncome']

le = LabelEncoder()
for obj in categorical_columns:
    train[obj] = le.fit_transform(train[obj])

print(train.dtypes)


print(train.keys())

def model_function (model_, dtrain, predictors, outcome):
    # model_.fit(dtrain[predictors], dtrain[outcome])
    # prediction = model_.predict(dtrain[predictors])
    #
    # #accuracy
    # accuracy = accuracy_score(prediction, dtrain[outcome])
    # cv_score = cross_val_score(model_, dtrain[predictors], dtrain[outcome], cv=5)
    # cv_score = np.mean(cv_score)
    #
    # # print(accuracy)
    # # print(cv_score)
    #
    # # Perform Cross-validation

    # X = dtrain[predictors]
    # y = dtrain[outcome]
    X = dtrain.drop('Loan_Status', 1)
    y = dtrain.Loan_Status

    X = pd.get_dummies(X)
    all_score = []
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    for train_index, test_index in kf.split(X, y):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        model.fit(xtrain, ytrain)
        pred_test = model.predict(xtest)
        score = accuracy_score(ytest, pred_test)
        print(score)


outcome_var = 'Loan_Status'
predictors_var = ['Credit_History', 'Gender', 'Education', 'EMI']

print('LR')
model_function(LogisticRegression(), train, predictors_var, outcome_var )
print('RFC')
# model_function(RandomForestClassifier(), train, predictors_var, outcome_var)
print('DTC')
# model_function(DecisionTreeClassifier(), train, predictors_var, outcome_var)
