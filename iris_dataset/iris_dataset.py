import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import mglearn
iris = load_iris()

print(iris.keys())
print(iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = KNeighborsRegressor(n_neighbors=1)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(model.score(X_test, y_test))

iris_dataframe = pd.DataFrame(data=X_train, columns=iris['feature_names'])
print(iris_dataframe)

# grr = pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize =(15, 15), marker = 'o', hist_kwds ={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

print(iris_dataframe.describe())