import numpy as np


class ClassificationModel(object):
    def __init__(self):
        pass

    def train_classifier(self, data,labels):
        """ Train classifier on labels and data

        Args:
            labels: a one dimensional vector of size N
            data: a two dimensional vector of size NxK
        Raises:
            AssertionError: if labels is not a one-dimensional matrix.
            AssertionError: if data is not a two-dimensional matrix.
            AssertionError: if labels and data have different length.

        """
        assert len(np.shape(labels)) == 1, ":labels should be a vector"
        assert len(np.shape(data)) == 2, ":data should be a matrix"
        assert np.shape(labels)[0] == np.shape(data)[0], "length of :labels should equal length of :data"
        
        self._train_classifier(labels, data)
        
    def classify_data(self, data):
        """Classifies unclassified data.

        Args:
            data: The data to be classified
        Returns:
            predicted_labels: The predicted labels
        Raises:
            AssertionError: if data is not a two-dimensional matrix.

        """

        assert len(np.shape(data)) == 2, "Data [data], should be a matrix"
        predicted_labels = self._classify_data(data)
        return predicted_labels

    def test_classifier(self, data, labels):
        """ Tests the performance of the classification model.

        Args:
            data: the data, a two dimensional vector of size NxK
            labels: the correct labels, a one dimensional vector of size N
        Returns:
            q: the ratio of correctly labeled data points
            predicted_labels: The predicted labels
        Raises:
            AssertionError: if labels is not a one-dimensional matrix.
            AssertionError: if data is not a two-dimensional matrix.
            AssertionError: if labels and data have different length.

        """
        assert len(np.shape(labels)) == 1, ":labels should be a vector"
        assert len(np.shape(data)) == 2, ":data should be a matrix"
        assert np.shape(labels)[0] == np.shape(data)[0], "length of :labels should equal length of :data"

        predicted_labels = self.classify_data(data)
        q = np.mean(predicted_labels == labels)
        return q, predicted_labels

    def _classify_data(self, data):
        """ SHOULD NOT BE CALLED DIRECTLY!

        Private method for classifying data to be overridden by subclasses.

        Args:
            data: the data, a two dimensional vector of size NxK
        Returns:
            predicted_labels: The predicted labels

        """
        assert False, "Should be subclassed"

    def _train_classifier(self, labels, data):
        """ SHOULD NOT BE CALLED DIRECTLY!

        Private method for training model on labeled data.

        Args:
            labels: a one dimensional vector of size N
            data: a two dimensional vector of size NxK

        """
        assert False, "Should be subclassed"






