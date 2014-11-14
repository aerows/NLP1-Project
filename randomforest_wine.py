import csv
import numpy as np
import randomForest as rf
from sklearn import ensemble
reload(rf)
train_data =np.loadtxt(open("datasets/winequality-red.csv","rb"),delimiter=";",skiprows=1)
model =  rf.trainRandomForest(train_data, 10, 0.6)


randomforest = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy')
randomforest.fit(train_data[:, 0:-1], train_data[:, -1])

correct_sklearn = 0
correct_ours = 0
predictions_sklearn = []
predictions_ours = []
for data_point in train_data:
    prediction_ours = rf.makeRandomForestPrediction(data_point,model)
    prediction_sklearn = randomforest.predict(data_point[0:-1])[0]
    predictions_sklearn.append(prediction_sklearn)
    predictions_ours.append(prediction_ours)
    # print prediction, data_point[-1]
    if data_point[-1] == prediction_sklearn:
        correct_sklearn += 1
    if data_point[-1] == prediction_ours:
        correct_ours += 1
print np.bincount(predictions_sklearn)
print np.bincount(predictions_ours)
print np.bincount((train_data[:,-1]).astype(int))
print "Correct:", correct_ours, "Total: ", len(train_data), "Percentage: ", float(correct_ours)/len(train_data)
print "Correct:", correct_sklearn, "Total: ", len(train_data), "Percentage: ", float(correct_sklearn)/len(train_data)