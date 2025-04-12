import numpy as np
from generate_classification_dataset import generate_classification_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, y = generate_classification_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


class LogisticRegression():

    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass



print("done")
