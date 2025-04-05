import numpy as np
import pandas as pd

# weak learner
from sklearn.tree import DecisionTreeRegressor

# sklearn docs for GB Regression
# subsample: float, default=1.0
# The fraction of samples to be used for fitting the individual base learners. 
# If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with 
# the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

class GradientBoostedTree():
    """
    GBT with sklearn.tree.DecisionTreeRegressor as base model
    loss is mse as a default
    """
    
    def __init__(self, num_learners=100, learning_rate=0.1, max_depth=3):
        self.trees = {}  # store trees 1,..,M inside here
        self.tree_weights = {}  # store the tree weights
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        for i=0, fit base model 
        then for m in 1:M-1 do
            - fit model to residuals of aggregation of m-1 models
            - calculate loss as a function of the weight of the m'th model L(y, f_(m-1) + w_m * h_m)
            - using line search, find optimal weight w_m
        """
        for i in range(self.num_learners):
            if i == 0:
                self.trees[i] = self._fit_base_learner(X, y)
                self.tree_weights[i] = 1
            else:
                # calc 'new' y --> for mse = y - fm()
                residuals = self._calc_residuals(X, y)
                self.trees[i] = self._fit_base_learner(X, residuals)
                pred_i = self.trees[i].predict(X)
                self.tree_weights[i] = self._calc_tree_weight(residuals=residuals, h=pred_i) * self.learning_rate
                

    
    def predict(self, X):
        """
        iterate thruogh all trees with corresponding weights and predict
        """
        preds = []
        for tree in self.trees.values():
            preds.append(tree.predict(X)) 
        preds_weighted = [w*pred for w, pred in zip(self.tree_weights.values(), preds)]  # debug this, as preds for each tree is a N x 1 vector
        preds_final = np.mean(preds_weighted, axis=0)
        #preds_final = preds_final / np.sum(list(self.tree_weights.values()))
        return preds_final


    def _calc_tree_weight(self, residuals, h):
        """
        TODO: actual implementation
        for now just equal weight
        """
        # optimize (y - fm - w * hm)**2 = (residuals - w * hm)**2
        w_closed = np.sum(h * residuals) / np.sum(h ** 2)
        return w_closed

    
    def _fit_base_learner(self, X, y):
        """
        default DecisionTree so no params for this fct
        """
        mod = DecisionTreeRegressor(max_depth=self.max_depth,
                                    min_samples_split=2, 
                                    min_samples_leaf=1, 
                                    min_weight_fraction_leaf=0.0)
        mod.fit(X, y)
        return mod
    

    def _calc_residuals(self, X, y):
        """
        can access current num of estimators via self.trees.keys()
        """
        preds = []
        for tree in self.trees.values():
            preds.append(tree.predict(X)) 
        # preds should be a list of length 'm' [self.tree.keys()] containing elems of length X.shape[0]
        preds_weighted = [w*pred for w, pred in zip(self.tree_weights.values(), preds)]  # debug this, as preds for each tree is a N x 1 vector
        preds_final = np.mean(preds_weighted, axis=0)
        #preds_final = preds_final / np.sum(list(self.tree_weights.values())
        
        return y - preds_final

