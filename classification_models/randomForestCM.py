from classification_model import ClassificationModel

import numpy as np
from sklearn import ensemble


class RandomForestCM(ClassificationModel):
    """ Random Forest Classifier

    Kwargs:
        n_estimators (int): Number of estimators (default=10)
        max_depth (int):    Max depth of individual trees (default=None)

    Attributes:
        random_forest (RandomForestClassifier): sklearn library's random forest classifier object
    """

    def __init__(self, n_estimators=10, max_depth=None):
        ClassificationModel.__init__(self)
        self.random_forest = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                             criterion='entropy',
                                                             max_depth=max_depth)

    def _train_classifier(self, labels, data):
        """ """
        self.random_forest.fit(data, labels)

    def _classify_data(self, data):
        """ """
        labels = self.random_forest.predict(data)
        return labels
