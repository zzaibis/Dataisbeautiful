import pandas as pd
import numpy as np
import missingno
import seaborn as sns
import matplotlib.pyplot as plt
import mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
titanic = pd.read_csv('train.csv')

pd.options.display.max_columns = titanic.shape[0]
#### UNIVARIATE ANALYSIS ####

categ = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
conti = ['Fare', 'Age']



# Distribution


fig = plt.figure(figsize=(30, 10))

for i in range(0, len(categ)):
    fig.add_subplot(3,3, i+1)
    sns.countplot(x = categ[i], data=titanic)

# plt.show()

j = 6
for col in conti:
    fig.add_subplot(3, 3, j + 1)
    sns.distplot(titanic[col].dropna());
    j += 1

#### BIVARIATE ANALYSIS ####
fig = plt.figure(figsize=(30, 10))
i = 1
for obj in categ:
    fig.add_subplot(3,3, i)
    sns.countplot(x=obj, data=titanic, hue='Survived')
    i += 1

fig.add_subplot(3,3,6)
sns.swarmplot(x="Survived", y="Age", hue="Sex", data=titanic);
# titanic.boxplot(column='Age', by='Survived' )
fig.add_subplot(3,3,7)
# sns.boxplot(x='Survived', y='Age', data=titanic)

# Fare  and Survived
fig.add_subplot(3, 3, 8)
# sns.boxplot(x='Survived', y='Fare', data=titanic)
# sns.boxplot(x='Survived', y='Fare', data=titanic)


# Correlation
corr = titanic.drop(['PassengerId'], axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)
# fig.add_subplot(3,3,9)
# sns.heatmap(corr, mask=mask, cmap=cmap, cbar_kws={"shrink": .5})

# plt.show()
# fig.clear()

# Feature Engineering

title = ['Mlle','Mrs', 'Mr', 'Miss','Master','Don','Rev','Dr','Mme','Ms','Major','Col','Capt','Countess']

def ExtractTitle(name):
    tit = 'missing'
    for item in title:
        if item in name:
            tit = item
    if tit == 'missing':
        tit = 'Mr'
    return tit

titanic['Title'] = titanic.apply(lambda x: ExtractTitle(x['Name']), axis=1)

plt.figure(figsize=(13, 5))
fig.add_subplot(2,1,1)
# sns.countplot(x='Title', hue='Survived', data = titanic)

# print(titanic.isnull().sum())

MedianAge = titanic['Age'].median()

print(MedianAge)

# titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

titanic['Age'] = titanic['Age'].fillna(MedianAge)

ModeEmbarked = titanic['Embarked'].mode()[0]
titanic['Embarked'] = titanic['Embarked'].fillna(ModeEmbarked)

# Cabin
titanic["Cabin"] = titanic.apply(lambda obs: "No" if pd.isnull(obs['Cabin']) else "Yes", axis=1)
titanic = pd.get_dummies(titanic, drop_first=True, columns=['Sex', 'Title', 'Cabin', 'Embarked'])

scale_transform = titanic[['Age', 'Fare']]

scale = StandardScaler().fit(scale_transform)
scale_transform = scale.transform(scale_transform)

titanic[['Age', 'Fare']] = scale_transform

target = titanic.Survived

features = titanic.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

MlRes = {}
def MlResult(model, score):
    MlRes[model] = score
    print(MlRes)


roc_curve_data = {}
def ConcatRocData(algoname, fpr, tpr, auc):
    data = [fpr, tpr, auc]
    roc_curve_data[algoname] = data

# Logictics Regression

lr = LogisticRegression()

print(lr.coef_)

# print(score)

def models_of_titanic(algo, xtrain, ytrain, xtest, ytest):
    algo.fit(xtrain, ytrain)
    algo.predict(xtest)
    score = algo.score(xtest, ytest)
    model_used = str(algo).split('(')[0]
    print(model_used, ':', score)
    if model_used in ['RandomForestClassifier', 'DecisionTreeClassifier']:
        feat_i = pd.Series(algo.feature_importances_, xtrain.columns).sort_values(ascending=False)
        feat_i.plot(kind = 'bar', title = 'feature importance')
    else:
        feat_i = pd.Series(algo.coef_, xtrain.columns).sort_values(ascending=False)
        feat_i.plot(kind='bar', title='feature coefficients')
    plt.show()
    plt.show()

# models_of_titanic(LogisticRegression(), X_train, y_train, X_test, y_test)
# models_of_titanic(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
# models_of_titanic(RandomForestClassifier(n_estimators=100), X_train, y_train, X_test, y_test)