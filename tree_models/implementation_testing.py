import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import helpers as hf

data = pd.read_csv(r"C:\Users\Damja\CODING_LOCAL\algorithms_from_scratch\data\house-prices-advanced-regression-techniques\train.csv")
train, test = train_test_split(data, test_size=0.3, shuffle=False, random_state=420)
#print(data.info())


# create random dataset
# first 5 coeffs are nonlinear, last 5 are linear
X, y, true_coeffs = hf.create_random_dataset(n_samples=4000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=420)


# my decision tree model
from simple_decision_tree import SimpleDecisionTreeRegressor
custom_dectree = SimpleDecisionTreeRegressor(max_depth=5)
custom_dectree.fit(X_train, y_train)

# base model simple regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
ridge = Ridge()
ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, preds)
print(ridge.coef_)


# sklearn decision tree model
from sklearn.tree import DecisionTreeRegressor
sk_dectree = DecisionTreeRegressor(max_depth=5)
sk_dectree.fit(X_train, y_train)
preds = sk_dectree.predict(X_test)
sk_dectree_mse = mean_squared_error(y_test, preds)




print("done")