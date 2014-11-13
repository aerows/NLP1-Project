import numpy as np
def hasMultipleTargetClasses(data):
    # Count the amount of unique target classes that are left in this node/leaf
    # If more than one, the node is still unpure (and would require another split if possible)
    return len(np.unique(data[:,-1])) > 1

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
    if hasMultipleTargetClasses(data)
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
    
def decisionStump(data):
    """Returns the optimal attribute and value for a tree to split on given
    some data matrix"""
    
    #find atrribute and value where entropy of upper and lower are smallest
    entropy = np.inf;
    #for all attrbutes:
    K = data.shape[1];
    for k_temp in range(K):
        #sort on values of that attribute
        sort = data[data[:,2].argsort()]
        #for all values
        for value_temp in np.unique(sort[:,2]):
            #calculate entropy of this split (add upper and lower)
            lower = data[data[:,k_temp]<=value_temp]
            upper = data[data[:,k_temp]>=value_temp]
            histlower ,_ = np.histogram(lower[:,-1],len(np.unique(lower[:,-1])));
            histupper, _ = np.histogram(upper[:,-1],len(np.unique(upper[:,-1])));
            entropylower = calculateEntropy(histlower);
            entropyupper = calculateEntropy(histupper);
            #if entropy is smaller 
            print (entropylower+ entropyupper)
            if entropylower+entropyupper < entropy:
                entropy = entropylower +entropyupper;
                k = k_temp;
                value = value_temp;
                print 'k: ', k, 'value: ', value, 'entropy: ', entropy;
                #save this attribute value pair as new optimal
    
    #random split
    #K = data.shape[1]
    #k = np.random.randint(K-1);
    #value = np.mean(data[:,k]);
    return k, value
    
def calculateEntropy(histogram):
    if len(np.unique(histogram)) <= 1:
        H = 0;
    else:
        H = 0;
        prob = np.double(histogram)/sum(np.double(histogram));
        for p in prob:
            H += -p*np.log2(p);
    return H