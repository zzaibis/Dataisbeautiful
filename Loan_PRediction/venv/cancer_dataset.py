from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()


# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(cancer['feature_names'])


X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], stratify=cancer['target'], random_state=42)

#
# # print(X_train.shape)
# # print(X_test.shape)
# # print(y_train.shape)
# # print(y_test.shape)
#
# knn = KNeighborsClassifier(n_neighbors=3)
#
# # print(knn.fit(X_train, y_train))
# #
# # print(knn.predict(X_test))
# #
# # # print(knn.score(X_train, y_train))
# #
# # print(knn.score(X_test, y_test))
#
# logreg = LogisticRegression(C=100)
#
#
# def classification_model(model, train_data, train_outcome, test_data, test_outcome):
#     model.fit(train_data, train_outcome)
#     A = model.predict(test_data)
#     B = model.score(train_data, train_outcome)
#     C = model.score(test_data, test_outcome)
#     print(A)
#     print(B)
#     print(C)
#
# classification_model(knn, X_train, y_train, X_test, y_test)
# classification_model(logreg, X_train, y_train, X_test, y_test)

tree = DecisionTreeClassifier(max_depth=4,random_state=0)

tree.fit(X_train, y_train)
print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)
