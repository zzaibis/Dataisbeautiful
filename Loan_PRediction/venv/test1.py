
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("train_loan.csv")


# print(df[['Gender', 'Married']])
pd.options.display.max_columns = df.shape[0]

# Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
#       dtype='object')

# grr = pd.scatter_matrix(df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] )

plt.show()

# print(df.describe())
# print(df["Gender"].value_counts())
# print(df.apply(lambda x: sum(x.isnull()),axis = 0))
# df.ApplicantIncome.hist(by=df["Gender"],bins=50)
# df.boxplot(column= "ApplicantIncome", by= "Education")
# plt.scatter(df.LoanAmount, df.Loan_Amount_Term)
# df.CoapplicantIncome.hist(bins=50)
# df["TotalIncome"] = df.ApplicantIncome + df.CoapplicantIncome
# df.TotalIncome.hist(bins=50)
# print(df.head())
# print(df.apply(lambda x: sum(x.isnull()), axis=0))
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(df["Gender"].fillna(df["Gender"].mode()[0], inplace=True))
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# print(df.apply(lambda x: sum(x.isnull()),axis=0))

# df.ApplicantIncome.hist(bins=50)
# df.ApplicantIncome.hist()
# df.ApplicantIncome.hist(bins=100)
# df.CoapplicantIncome.hist(bins=50)
# TotalIncome = df.ApplicantIncome + df.CoapplicantIncome
# df = pd.DataFrame(df, index=TotalIncome)
# df["TotalIncome"] = df.ApplicantIncome + df.CoapplicantIncome
# print(df.head())

# TotalIncome.hist(bins=50)
# df.boxplot(column='ApplicantIncome', by=["Property_Area"])
# df["ApplicantIncome_log"] = np.log(df["ApplicantIncome"])
# df["ApplicantIncome_log"].hist(bins=50)
# df.boxplot(column="ApplicantIncome_log")
# print(df.apply(lambda x: sum(x.isnull()), axis=0))
# pivot_data = df.pivot_table(values=["ApplicantIncome"], index=[ "Property_Area", "Education"], columns=  "Gender", aggfunc="count")
# graduate_females_from_rural = df.loc[(df.Gender == "Female") & (df.Education == "Graduate") & (df.Property_Area == "Rural")]
# print(graduate_females_from_rural.head())
# print(df.apply(lambda x: sum(x.isnull()), axis=0))
# print(pivot_data)
# print(df.pivot_table(values= "ApplicantIncome" , index= ["Gender", "Education"], columns= ["Property_Area"], aggfunc= "mean"))
# print(df.loc[(df.Gender == "Female") & (df.Education == "Graduate")].count())

# temp1 = df['Credit_History'].value_counts(ascending=True)
# temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
# print ('Frequency Table for Credit History:')
# print (temp1)
#
# print ('\nProbility of getting loan for each Credit History class:')
# print (temp2)
#
# fig = plt.figure(figsize=(10,4))
# ax1 = fig.add_subplot(121)
# ax1.set_xlabel('Credit_History')
# ax1.set_ylabel('Count of Applicants')
# ax1.set_title("Applicants by Credit_History")
# temp1.plot(kind='bar')
#
# ax2 = fig.add_subplot(122)
#
# ax2.set_xlabel('Credit_History')
# ax2.set_ylabel('Probability of getting loan')
# ax2.set_title("Probability of getting loan by credit history")
# temp2.plot(kind = 'bar')

# temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
# temp4 = df.pivot_table(values= "Credit_History", columns="Loan_Status", aggfunc= "count" )
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(temp4)
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,")
# print(temp3)
#
# print(df["LoanAmount"].mean())
#
# print(df.apply(lambda x: sum(x.isnull()), axis=0))
# df["LoanAmount"].fillna(df["LoanAmount"].mode(), inplace=True)
# print(df.apply(lambda x: sum(x.isnull()), axis=0))

# df.boxplot(column="ApplicantIncome", by= ["Gender", "Education"])
# print(df.loc[(df["Gender"] == "Female") & (df["ApplicantIncome"] >= 18000)])
# print(df.loc[(df["Gender"] == "Male") & (df["ApplicantIncome"] <= 20000) & (df["ApplicantIncome"] >= 18000) ])
#
# print(df["Gender"].value_counts())
# loan_yn = pd.crosstab(df["Credit_History"], df["Loan_Status"])
# loan_yn.plot(kind='bar', stacked=False, color=['red', 'blue'], grid=False)

# print(df.apply(lambda x: sum(x.isnull()), axis=0))

df["Self_Employed"].fillna("No", inplace=True)
df["Self_Employed"].fillna("No", inplace=True)



table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# table2 = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=["count"])
# print(table)


# print(df.LoanAmount.isna().sum())
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]



df["LoanAmount_log"] = np.log(df["LoanAmount"])

# print(df.head())

# df["LoanAmount_log"].hist(bins=25)

df["TotalIncome_log"] = np.log(df["TotalIncome"])

df["TotalIncome_log"].hist(bins=25)

df["Paybackrate"] = (df["LoanAmount"] /df["TotalIncome"])*100
# print(df["Paybackrate"].max())
#
# print(df.loc[(df["Paybackrate"] == df["Paybackrate"].max())])
# print(df.head())
# plt.show()

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# print(df.apply(lambda x: sum(x.isnull()), axis=0))
#

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status', 'TotalIncome', 'LoanAmount_log', 'TotalIncome_log', 'Paybackrate']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes


# print(df['Credit_History'])

outcome_var = 'Loan_Status'
lr = LogisticRegression()
dataset = df
predictor_var = ['Credit_History']

def classification_model(model,data,predictors, outcome):
    # Fit the model
    model.fit(data[predictors], data[outcome].values.ravel())
    predictions = model.predict(data[predictors])
    # print(predictions)
    accuracy = metrics.accuracy_score(predictions, data[outcome].values.ravel())
    print(accuracy)
    # Cross-validation
    scores = cross_val_score(model, data[predictors], data[outcome].values.ravel(), cv=5)
    scores_kfold = cross_val_score(model, data[predictors], data[outcome].values.ravel(), cv=kfold)
    print(np.mean(scores))
    print(np.mean(scores_kfold))


classification_model(lr, dataset, predictor_var, outcome_var)
#     # Fit the model
#     model.fit(predictor_var, outcome_var.values.ravel())
#
#     # Make predictions on training set
#     predictions = model.predict(predictor_var)
#
#     # Print Accuracy
#     accuracy = metrics.accuracy_score(predictions, outcome_var.values.ravel())
#     print(accuracy)
#
#     # Cross-validation using cross_val_score
#     scores = cross_val_score(model, predictor_var, outcome_var.values.ravel(), cv=5)
#     print(np.mean(scores))
#     scores_kfold = cross_val_score(model, predictor_var, outcome_var.values.ravel(), cv=kfold)
#     print(np.mean(scores_kfold))



#
#
#
#     # Fit the model again so that it can be refered outside the function:
#     model.fit(data[predictors], data[outcome])
#
# Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status',
#        'TotalIncome', 'LoanAmount_log', 'TotalIncome_log', 'Paybackrate'],

#
# outcome_var = 'Loan_Status'
# model = LogisticRegression()
# data = df
# predictor_var = ['Credit_History']
#
# # Logicticregrression Predictive Model
# # lr = LogisticRegression()
#
# # classification_model(model, df,predictor_var,outcome_var)
# model.fit(data[predictor_var], data[outcome_var].values.ravel())
# predictions = model.predict(data[predictor_var])
# # print(predictions)
# accuracy = metrics.accuracy_score(predictions, data[outcome_var].values.ravel())
# print(accuracy)
# scores = cross_val_score(model, data[predictor_var], data[outcome_var].values.ravel(), cv=5)
#
# # kfold = KFold(n_splits=5)
# scores_kfold = cross_val_score(model, data[predictor_var], data[outcome_var].values.ravel(), cv=kfold)
# # print(scores)
# print(np.mean(scores))
# print(np.mean(scores_kfold))
# print(scores.mean())