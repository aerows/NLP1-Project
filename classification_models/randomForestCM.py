from classification_model import ClassificationModel

import numpy as np
from sklearn import ensemble


class RandomForestCM(ClassificationModel):
    def __init__(self):
        ClassificationModel.__init__(self)
        self.random_forest = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy')

    def _train_classifier(self, T, D):
        """ Should be subclassed """
        self.random_forest.fit(D, T)

    def _classify_data(self, D):
        """ Should be subclassed """
        T = self.random_forest.predict(D)
        return T
