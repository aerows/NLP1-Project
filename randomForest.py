import numpy as np

def trainRandomForest(traindata, Ntrees):
    """Creates a random forest model from the data"""

    # traindata: A matrix of all the data (NxK)
    # Ntrees: the number of trees that make up the forest

    model = []#a list of all trees

    for n in range(Ntrees):
    
        #bootstrap data
        #set some number of features
        #set some max depth
        tree = trainDecisionTree(bootStrapData, maxdepth) #train
        model[n] = tree
    
    return model

def trainDecisionTree(data, depth=0, tree = []):
    """Returns the decision tree for some data"""
    #find optimal split
    if len(np.unique(data[:,-1])) > 1:
        d = depth
        splitFeature, splitValue = decisionStump(data)
        
        lower = data[data[:,splitFeature]<=splitValue]
        upper = data[data[:,splitFeature]>splitValue]
        
        tree_lower = trainDecisionTree(lower, depth=d+1, tree = tree)
        tree_upper = trainDecisionTree(upper, depth=d+1, tree = tree)
        
        tree = tree.append([[splitFeature,splitValue],tree_lower,tree_upper]);
        
        #see if child nodes are "pure"
        #for all not pure and depth<max repeat
    
        return tree
    else:
        return tree
    
def decisionStump(data):
    K = data.shape[1]
    k = np.random.randint(K-1);
    value = np.mean(data[:,k]);
    return k, value