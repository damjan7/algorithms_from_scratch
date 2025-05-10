import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import helpers as hf
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv(r"C:\Users\Damja\CODING_LOCAL\algorithms_from_scratch\data\house-prices-advanced-regression-techniques\train.csv")
train, test = train_test_split(data, test_size=0.3, shuffle=False, random_state=420)
#print(data.info())


# create random dataset
# first 5 coeffs are nonlinear, last 5 are linear
X, y, true_coeffs = hf.create_random_dataset(n_samples=4000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=420)


########################
from sklearn.tree import DecisionTreeRegressor
# single decision tree
from simple_decision_tree import SimpleDecisionTreeRegressor
mod = DecisionTreeRegressor(max_depth=3,
                                    min_samples_split=2, 
                                    min_samples_leaf=1, 
                                    min_weight_fraction_leaf=0.0)
mod.fit(X_train, y_train)
preds_mod = mod.predict(X_test)
print("mse single dec tree, max_depth 3: ", mean_squared_error(preds_mod, y_test))

# TEST GRADIENT BOSTED TREES
from gradient_boosted_tree import GradientBoostedTree
my_gbt = GradientBoostedTree()
my_gbt.fit(X_train, y_train, x_val=X_test, y_val=y_test)
preds_my_gbt = my_gbt.predict(X_test)
print("mse my gbt: ", mean_squared_error(preds_my_gbt, y_test))

from gradient_boosted_tree_v2 import GradientBoostedTree
my_gbt_v2 = GradientBoostedTree()
my_gbt_v2.fit(X_train, y_train, x_val=X_test, y_val=y_test)
preds_my_gbt_v2 = my_gbt_v2.predict(X_test)
print("mse my gbt v2: ", mean_squared_error(preds_my_gbt_v2, y_test))

from sklearn.ensemble import GradientBoostingRegressor
sk_gbt = GradientBoostingRegressor()
sk_gbt.fit(X_train, y_train)
preds_sk_gbt = sk_gbt.predict(X_test)
print("mse sklearn gbt: ", mean_squared_error(preds_sk_gbt, y_test))

print("done")
# DONE
########################



# my decision tree model
from simple_decision_tree import SimpleDecisionTreeRegressor
custom_dectree = SimpleDecisionTreeRegressor(max_depth=5)
custom_dectree.fit(X_train, y_train)
preds_custom_dectree = custom_dectree.predict(X_test)
# constantly predicts custom_dectree.tree[2][3][2][3][5]

# sklearn decision tree model
from sklearn.tree import DecisionTreeRegressor
sk_dectree = DecisionTreeRegressor(max_depth=5)
sk_dectree.fit(X_train, y_train)
preds_sklearn_dectree = sk_dectree.predict(X_test) 
sk_dectree_mse = mean_squared_error(y_test, preds)


# base model simple regression
from sklearn.linear_model import Ridge
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