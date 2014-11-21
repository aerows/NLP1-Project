import numpy as np

class ClassificationModel:
    def __init__(self):
        pass

    def train_classifier(self, T, D):
        """ Train classifier on T and D
        
	    Arguments:
	       T -- the labels, a one dimensional vector of size N
	       D -- the data, a two dimensionel vector of size NxK
	"""
        assert len(np.shape(T)) == 1, "Labels [T] should be a vector"
        assert len(np.shape(D)) == 2, "Data [D], should be a matrix"
        assert np.shape(T)[0] == np.shape(D)[0], "Number of Lables [T] should equal number of Data [D]"
        
        self._train_classifier(T,D)
        
    def classifiy(self, D):        
        """ Classifiy data D

	   Arguments:
	       D -- the data, a two dimensionel vector of size NxK
	       
	   Return:
	       T -- the predicted labels, a one dimensional vector of size N
	"""
        assert len(np.shape(D)) == 2, "Data [D], should be a matrix"
        return self._classify(D)

# To be subclassed

    def _classify(self, D):
        """ Should be subclassed """
        print "Should be subclassed"
        return None
    
    def _train_classifier(self, T, D):
        """ Should be subclassed """
        print "Should be subclassed"
        
        
       
        
    