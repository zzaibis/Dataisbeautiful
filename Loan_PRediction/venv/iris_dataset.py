from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()

# print(iris_dataset["data"])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset['target'],random_state=0)


knn = KNeighborsClassifier(n_neighbors=1)

print(knn.fit(X_train, y_train))

print(knn.predict(X_test))

# print(knn.score(X_train, y_train))

print(knn.score(X_test, y_test))

