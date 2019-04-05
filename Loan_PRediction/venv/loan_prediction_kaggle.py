import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# Reading the Datasets
train = pd.read_csv("train_loan.csv")

test = pd.read_csv("test_loan.csv")

pd.options.display.max_columns = train.shape[0]

# Checking how each categorical variable says about the data

# print("Loan Status:\n{}".format(train['Loan_Status'].value_counts(dropna=False)))
#
# # train['Loan_Status'].value_counts(dropna=False).plot.bar(title= "Loan Status")
#
# print("Gender:\n{}".format(train['Gender'].value_counts(dropna=False)))
#
# print("Married:\n{}".format(train['Married'].value_counts(dropna=False)))
#
# print("Dependents:\n{}".format(train['Dependents'].value_counts(dropna=False)))
#
# print("Education:\n{}".format(train['Education'].value_counts(dropna=False)))
#
# print("Self_Employed:\n{}".format(train['Self_Employed'].value_counts(dropna=False)))
# Percentage who has credit history and got loan
# print("Credit History:\n{}".format(train['Credit_History'].value_counts(normalize=True)))
# probabilty of Yes and No who got Loan
# temp2 = train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
# print(temp2)
#
# print("Property Area:\n{}".format(train['Property_Area'].value_counts(dropna=False)))
#
# train.boxplot(column='ApplicantIncome', by= 'Education')
#
# train['ApplicantIncome'].plot.box()
# df = train.dropna()
# sns.distplot(df['LoanAmount'])

# print(train.isnull().sum())




# #############################Categorical Independent Variable vs Target Variable #################################################

# Gender = pd.crosstab(train['Gender'], train['Loan_Status'])

# print(Gender)

# Gender.plot.bar()
# Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# Gender.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
#
#
# Education = pd.crosstab(train['Education'], train['Loan_Status'])
# print(Education)
# Education.plot(kind='bar', stacked=True, color=['green', 'black'])
#
# Married = pd.crosstab(index=train['Married'], columns=train['Loan_Status'])
# print(Married)
# Married.plot(kind='bar', stacked= True, color = ['pink', 'orange'] )
#
#
# Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
# print(Dependents)
# Dependents.plot(kind='bar', stacked= True, color= ['magenta', 'red'])
#
# Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
# print(Property_Area)
# Property_Area.plot(kind= 'bar', stacked= True, color= ['black', 'purple'])
#
# Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
# print(Self_Employed)
# Self_Employed.plot(kind='bar', stacked =True, color= ['blue', 'violet'])
#
# Credit_history = pd.crosstab(train['Credit_History'], train['Loan_Status'])
# print(Credit_history)
# Credit_history.plot(kind='bar', stacked=True, color=['grey', 'green'])

# matrix = train.corr()
# print(matrix)

# f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(matrix, vmax=.8, square=False, cmap="BuPu");
# plt.show()
################################################################ Filling na values and some other adjustments



# For Train Set
Gender_mode = train['Gender'].mode()[0]
train['Gender'].fillna(Gender_mode, inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train["Credit_History"].mode()[0], inplace=True)
train['Dependents'].replace('3+', 3,inplace=True)
print(train.isnull().sum())

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
print(test.isnull().sum())



# train['Loan_Status'].replace('N', 0,inplace=True)
# train['Loan_Status'].replace('Y', 1,inplace=True)


'''
X = train.drop('Loan_Status', axis=1)
y = train.Loan_Status


X=pd.get_dummies(X)

train=pd.get_dummies(train)
test = pd.get_dummies(test)

x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

print(x_train.shape)
print(x_cv.shape)

lr = LogisticRegression()

print(lr.fit(x_train, y_train))

pred_cv = lr.predict(x_cv)

# print(pred_cv)

print(lr.score(x_cv, y_cv))
print(accuracy_score(y_cv, pred_cv))


#PRedicting on test set
pred_test = lr.predict(test)
#
# print(pred_test)

# print(lr.score(pred_test, y_cv))

# Logistic Regression using StratifiedkFOld

print('******************************************************************************************************************************')
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    # print(train_index)
    # print(kf)
    # print(test_index)
    print('\n{} of kfold {}'.format(i, kf.n_splits))


    xtrain, xtest = X.loc[train_index], X.loc[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    model = LogisticRegression(random_state=1)
    model.fit(xtrain, ytrain)
    pred_test = model.predict(xtest)
    score = accuracy_score(ytest, pred_test)
    print('accuracy_score', score)
    i += 1


pred_test_k = model.predict(test)
pred=model.predict_proba(xtest)[:,1]
'''

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis =1)
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']

train['TotalIncome_log'] = np.log(train['TotalIncome'])
test['TotalIncome_log'] = np.log(test['TotalIncome'])

train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term']

train['Balance_Income'] = train['TotalIncome'] - (train['EMI']*1000)
test['Balance_Income'] = test['TotalIncome'] - (test['EMI']*1000)

train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
#
# print(train.head())
# X = train.drop('Loan_Status', 1)
# y = train.Loan_Status
#
# X=pd.get_dummies(X)

# train=pd.get_dummies(train)
# test = pd.get_dummies(test)


# i = 1
# kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
# for train_index, test_index in kf.split(X, y):
#     # print(train_index)
#     # print(kf)
#     # print(test_index)
#     print('\n{} of kfold {}'.format(i, kf.n_splits))
#
#
#     xtrain, xtest = X.loc[train_index], X.loc[test_index]
#     ytrain, ytest = y[train_index], y[test_index]
#
#     model = LogisticRegression(random_state=1)
#     model.fit(xtrain, ytrain)
#     pred_test = model.predict(xtest)
#     score = accuracy_score(ytest, pred_test)
#     print('accuracy_score', score)
#     i += 1
#
def loan_prediction_fn(train_set, model):
    X = train_set.drop('Loan_Status', 1)
    y = train_set.Loan_Status

    X = pd.get_dummies(X)

    i = 1
    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        xtrain, xtest = X.loc[train_index], X.loc[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        model.fit(xtrain, ytrain)
        pred_test = model.predict(xtest)
        score = accuracy_score(ytest, pred_test)
        print('accuracy_score', score)
        i += 1

loan_prediction_fn(train, LogisticRegression() )

'''
print("################## Decision Tree CLassifier #############################")



from sklearn.tree import DecisionTreeClassifier

i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    # print(train_index)
    # print(kf)
    # print(test_index)
    print('\n{} of kfold {}'.format(i, kf.n_splits))


    xtrain, xtest = X.loc[train_index], X.loc[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    model = DecisionTreeClassifier(random_state=1)
    model.fit(xtrain, ytrain)
    pred_test = model.predict(xtest)
    score = accuracy_score(ytest, pred_test)
    print('accuracy_score', score)
    i += 1



plt.show()

'''