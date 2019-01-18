from .model import Model
from sklearn.neural_network import MLPClassifier
import numpy as np


class MLPsk(Model):

    def __init__(self,dist):
        super().__init__(dist)
        self.model = MLPClassifier((100,100,100),activation='tanh',
                        early_stopping=True, tol=1e-6,
                        n_iter_no_change=10,learning_rate_init=1e-2)

    def train(self, data, labels, verbose=False):
        self.model.verbose = verbose
        self.model.fit(data,labels)

    def predict(self, data):
        return self.model.predict_proba(data)