import numpy as np

def hashfeatures(baby, B, FIX):
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v

def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X

def genTrainFeatures(file1,file2,dimension=128, fix=3):
    """
    function [x,y]=genTrainFeatures
    
    This function calls the python script "name2features.py" 
    to convert names into feature vectors and loads in the training data. 
    
    
    Output: 
    x: n feature vectors of dimensionality d [d,n]
    y: n labels (-1 = girl, +1 = boy)
    """
    
    # Load in the data
    Xgirls = name2features(file1, B=dimension, FIX=fix)
    Xboys = name2features(file2, B=dimension, FIX=fix)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]