from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
boston = load_boston()
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
pd.options.display.max_columns = boston_df.shape[0]
boston_df['PRICE'] = boston.target


# print(boston_df)

prices = boston_df['PRICE']

features = boston_df.drop('PRICE', axis=1)
predictor_var = [x for x in boston_df.keys()]
predictor_var = [x for x in predictor_var if x not in ['PRICE']]
# predictor_var = [x for x in predictor_var if x in ['RM', 'LSTAT', 'PTRATIO']]

predictor_var = ['RM', 'LSTAT', 'DIS', 'NOX']
plt.figure(figsize=(20,5))

# print(features.columns)

# for i, col in enumerate(predictor_var):
#     plt.subplot(1, 3, i+1)
#     print(i, col)
#     plt.plot(boston_df[col], prices, 'o')
#     plt.plot(np.unique(boston_df[col]), np.poly1d(np.polyfit(boston_df[col], prices, 1))(np.unique(boston_df[col])))
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel('prices')
#

# print(boston_df['PRICE'].describe())
# print(boston_df.head())
# plt.show()


def boston_predict(model, dtrain, predictors, outcome):
    model.fit(dtrain[predictors], dtrain[outcome])
    prediction = model.predict(dtrain[predictors])
    # score = accuracy_score(dtrain[outcome], prediction)
    cv_score = cross_val_score(model, dtrain[predictors], dtrain[outcome], cv=6)
    # print(model.)
    print(cv_score.mean())
    print(pd.Series(model.feature_importances_, predictor_var).sort_values())


boston_predict(DecisionTreeRegressor(min_samples_split=20), boston_df, predictor_var, 'PRICE')
