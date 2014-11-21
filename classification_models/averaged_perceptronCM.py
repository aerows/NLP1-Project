# -*- coding: utf-8 -*-
from classification_model import ClassificationModel

import numpy as np

class AveragedPerceptronCM(ClassificationModel):
    def __init__(self):
        ClassificationModel.__init__(self)
        self.w = []
        self.b = 0

    def _train_classifier(self, T, D):
        itterations = 100
        
        # initilize w as a size k vector with zeros
        N,K = np.shape(D)
        w = np.zeros(K)
        b = 0.
        u = np.zeros(K)
        B = 0.
        # initialize b to zero
        # initilize u as a size k vector with zeros
        # initialize B to zero
        c = 1        
        for _ in range(itterations):
        #   for t,d in T,D do
            for i in range(N):
                t = T[i]
                d = D[i,:]
        #       if (t(w * x + b) ≤ 0 then
                if t*(w.dot(d.T) + b) <= 0:
        #           w = w + y * x
                    w = w + t * d.T
        #           b = b + y
                    b = b + t
        #           u = u + y * c * x
                    u = u + t * c * d
        #           B = B + y * c
                    B = B + t * c
            #   c++
                c += 1
        #  return w - (1/N) * u, b - (1/N) * B
        self.w = w - (1/N) * u
        self.b = b - (1/N) * B
        return self.w, self.b
        
    def _classify(self, D):
        T = np.sign(D.dot(self.w.T) + self.b)
        return T
