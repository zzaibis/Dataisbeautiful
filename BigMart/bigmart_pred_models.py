import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score

train = pd.read_csv('bigmart_train.csv')
test = pd.read_csv('bigmart_test.csv')

#Export submission file
# base1.to_csv("SampleSubmission.csv",index=False)

target_var = 'Item_Outlet_Sales'
IDCol = ['Item_Identifier', 'Outlet_Identifier']


# Model Pprediction Function
def prediction_model(model, dtrain, dtest, predictor, target,IDCol ,filename):
    # Fit the model
    model.fit(dtrain[predictor], dtrain[target])

    # Predict for training set
    dtrain_pred = model.predict(dtrain[predictor])

    # Performing cross validation

    cross_v_score = cross_val_score(model, dtrain[predictor], dtrain[target], cv= 20, scoring= 'neg_mean_squared_error')
    # cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20,scoring='mean_squared_error')

    cv_score = np.sqrt(np.abs(cross_v_score))

    # Calculating RMSE
    RMSE = np.sqrt(metrics.mean_squared_error(dtrain[target], dtrain_pred))

    print('RMSE : {}'.format(RMSE))

    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on Testing data
    dtest[target] = model.predict(dtest[predictor])

    # Export the calculated dtest values to submission file

    IDCol.append(target)

    submission = pd.DataFrame({x : dtest[x] for x in IDCol})

    submission.to_csv(filename, index=False)


##############################################################################################################################################
# 1
# Linear Regression Function

from sklearn.linear_model import LinearRegression, Lasso

predictors = [x for x in train.columns if x not in [target_var]+IDCol]

algo = LinearRegression(normalize=True)

prediction_model(algo, train, test, predictors, target_var, IDCol,"SubmissionLR.csv" )

# coefficients = pd.Series(algo.coef_, predictors).sort_values()

# print(coefficients)

# coefficients.plot(kind='bar', title ='Model Coefficients')

# plt.show()
################################################################################################################################################
# 2
# Ridge Linear Regression

algo1 = Lasso(alpha=0.05, normalize=True)

prediction_model(algo1, train, test, predictors, target_var, IDCol, "SubmissionRidge.csv")


# coefficients = pd.Series(algo1.coef_, predictors).sort_values()

# coefficients.plot(kind= 'bar', title = 'Model COefficients for Ridge Regression')

# plt.show()

#################################################################################################################################################
# 3
# Lasso Linear Regression

algo2 = Lasso(alpha=0.05, normalize=True)

prediction_model(algo2, train, test, predictors, target_var, IDCol, "SubmissionLasso.csv")


# coefficients = pd.Series(algo1.coef_, predictors).sort_values()

# coefficients.plot(kind= 'bar', title = 'Model COefficients for Ridge Regression')

# plt.show()


# predictors = ['Item_MRP','Outlet_Type_0','Outlet_5']

#################################################################################################################################################
# 4
# Decision Tree with all features

from sklearn.tree import DecisionTreeRegressor

algo3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

prediction_model(algo3, train, test, predictors, target_var, IDCol, 'SubmissionVTree.csv')

# coefficients = pd.Series(algo2.feature_importances_, predictors).sort_values(ascending=False)

# coefficients.plot(kind='bar', title= 'Feature Importance Using Decision Tree')

# plt.show()
#
# # This Decision Tree Visualisation code hangs the system

###################################################################################################################################################
# 5
# Decision Tree with four features

from sklearn.tree import DecisionTreeRegressor


algo4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=50)

prediction_model(algo4, train, test, predictors, target_var, IDCol, 'SubmissionVTree5.csv')



# coefficients = pd.Series(algo2.feature_importances_, predictors).sort_values(ascending=False)

# coefficients.plot(kind='bar', title= 'Feature Importance Using Decision Tree')

# plt.show()
#

#################################################################################################

# Random Forest
from sklearn.ensemble import RandomForestRegressor

algo5 = RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=100, n_jobs=4)

prediction_model(algo5, train, test, predictors, target_var, IDCol,'SubmissionRF.csv')

''''
from sklearn.tree import export_graphviz

# Visualize data
dot_data = export_graphviz(algo2,
                                feature_names=predictors,
                                class_names=target_var,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('max_features_5.png')

print(train.shape[1])

'''