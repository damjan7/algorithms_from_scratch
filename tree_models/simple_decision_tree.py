import numpy as np

"""
Implementation of a simple decision tree model
Input: Data is a 2d np array (samples, features)
Output: A simple decision tree model
"""

class SimpleDecisionTreeRegressor:

    def __init__(self, max_depth=5):
        self.max_depth = 5
        self.tree = None

    def fit(self, X, y):
        # assert X, y np arrays
        # assert shapes match
        self.tree = self._build_tree(X, y)
        
    def predict(self, X):
        """
        need to traverse the tree
        the tree is a nested tuple of (split feature, split index, left subtree, right subtree)
        where left and right subtree contain the same structure until a leaf node is reached
        in that case the left subtree and right subtree are None 
        """
        pass
    
    def _build_tree(self, X, y, depth=0):
        """
        tree building process:

        while depth not reached:
            1. find best split feature and split point
            2. split data in two halves
            3. step 1
        
        if depth reached:
            return 
            
        """
        
        if depth == 5 or X.shape[0]==1:
            return

        # find best two regions and split data accordingly
        split_feature, split_index = self._find_best_split(X, y)

        left_bool = X[:, split_feature] <= X[split_index, split_feature]
        right_bool = X[:, split_feature] > X[split_index, split_feature]

        X_left, y_left = X[left_bool, :], y[left_bool]
        X_right, y_right = X[right_bool, :], y[right_bool]

        # call build tree recursively
        left_subtree = self._build_tree(X_left, y_left, depth=depth+1)
        right_subtree = self._build_tree(X_right, y_right, depth=depth+1)

        return (split_feature, split_index, left_subtree, right_subtree)


    def _find_best_split(self, X, y):
        """
        iterate through all samples to find best split
        is "easy", since we just fit the avg (in case of MSE loss)
        in each of the regions after splitting
    
        """
        num_samples, num_features = X.shape
        best_split_feature, best_split_index = None, None
        information_gain = np.inf

        for feature in range(num_features):
            for split_index in range(0, num_samples):
                current_gain = self._information_gain(X, y, feature, split_index)
                if current_gain < information_gain:
                    information_gain = current_gain
                    best_split_feature, best_split_index = feature, split_index
        
        return best_split_feature, best_split_index

    
    def _information_gain(self, X, y, feature, split_index):
        """
        calculate information mse in both regions
        sum mse
        take negative mse sum
        """
        left_bool = X[:, feature] <= X[split_index, feature]
        right_bool = X[:, feature] > X[split_index, feature]
        y_left =  y[left_bool]
        y_right = y[right_bool]

        y_preds_left = np.mean(y_left)
        y_preds_right = np.mean(y_right)

        mse_left = self._mse(y_left, y_preds_left)
        mse_right = self._mse(y_right, y_preds_right)

        return -1 * (mse_left + mse_right)


    def _mse(self, y, y_preds):
        # assert y.shape[0] == len(y_preds), "y and y_preds are not same shape!"  doesn't make sense dumbass, ypreds is just 
        return ((y - y_preds)**2).mean()


    def _prune():
        pass