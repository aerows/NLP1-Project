from sklearn import ensemble
import pickle

# Getting back the objects:
with open('word_freq.pickle') as f:
    train_features, test_features, word_freq_fe, train_author_ids,test_author_ids = pickle.load(f)

randomforest = ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy')
randomforest.fit(train_features,train_author_ids)

correct = 0
for index,data_point in enumerate(test_features):
    prediction = randomforest.predict(data_point)
    print prediction, test_author_ids[index]
    if prediction == test_author_ids[index]:
        correct += 1

print correct, len(test_author_ids)