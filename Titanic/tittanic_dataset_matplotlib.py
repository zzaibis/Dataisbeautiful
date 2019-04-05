import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

titanic = pd.read_csv('train.csv')

pd.options.display.max_columns = titanic.shape[0]

print(titanic.keys())

# missingno.matrix(titanic)
#### UNIVARIATE ANALYSIS ####

categ = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
conti = ['Fare', 'Age']

# Distribution

fig = plt.figure(figsize=(30, 10))
fig2 = plt.figure(figsize=(30, 10))
for i in range(0, len(categ)):
    fig.add_subplot(3,3, i+1)
    fig2.add_subplot(3, 3, i + 1)
    # sns.countplot(x = categ[i], data=titanic)
    counts = titanic[categ[i]].value_counts(dropna=False)
    counts.plot(kind='bar',)
#
plt.show()