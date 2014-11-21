from classification_model import ClassificationModel

import numpy as np
from sklearn import ensemble


class RandomForestCM(ClassificationModel):
    def __init__(self):
        ClassificationModel.__init__(self)
        self.randomforest = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy')

    def _train_classifier(self, T, D):
        """ Should be subclassed """
        self.randomforest.fit(D, T)

    def _classify(self, D):
        """ Should be subclassed """
        T = self.randomforest.predict(D)
        return T
