from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df = pd.read_csv('housing.csv')
pd.options.display.max_columns = boston_df.shape[0]



prices = boston_df['MEDV']

features = boston_df.drop('MEDV', axis=1)
predictor_var = [x for x in boston_df.keys()]
predictor_var = [x for x in predictor_var if x not in ['MEDV']]

# plt.figure(figsize=(20,5))

print(features.columns)

X = features
y = prices


cv = ShuffleSplit(n_splits=10, test_size = 0.2, random_state = 0)
print(cv)
train_sizes = np.rint(np.linspace(1, X.shape[0] * 0.8 - 1, 9)).astype(int)
print(train_sizes)
# for i, col in enumerate(predictor_var):
#     plt.subplot(1, 3, i+1)
#     print(i, col)
#     plt.plot(boston_df[col], prices, 'o')
#     plt.plot(np.unique(boston_df[col]), np.poly1d(np.polyfit(boston_df[col], prices, 1))(np.unique(boston_df[col])))
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel('prices')

# plt.show()

def ModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    # Generate the training set sizes increasing by 50
    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0] * 0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10, 8))

    # Create three different models based on max_depth
    for k, depth in enumerate([1, 3,4, 5,6, 10]):
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth=depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = learning_curve(regressor, X, y, cv=cv, train_sizes=train_sizes, scoring='r2')
        # print(sizes)

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis=1)
        train_mean = np.mean(train_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 3, k + 1)
        ax.plot(sizes, train_mean, 'o-', color='r', label='Training Score')
        ax.plot(sizes, test_mean, 'o-', color='g', label='Testing Score')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')

        # Labels
        ax.set_title('max_depth = %s' % (depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0] * 0.8])
        ax.set_ylim([-0.5, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(2.05, 2.05), borderaxespad=0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=16, y=1.03)
    fig.tight_layout()
    fig.show()


def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1, 11)

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), X, y, param_name="max_depth", param_range=max_depth, cv=cv, scoring='r2')


    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(max_depth, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')

    # Visual aesthetics
    plt.legend(loc='lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05, 1.05])
    plt.show()

# ModelLearning(features, prices)
ModelComplexity(features, prices)
plt.show()