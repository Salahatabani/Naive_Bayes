import numpy as np
import sys, msvcrt
from Feature_hashing import *

def naivebayesPY(x,y):
    """
    function [pos,neg] = naivebayesPY(x,y);

    Computation of P(Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)

    Output:
    pos: probability p(y=1)
    neg: probability p(y=-1)
    """
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    y = np.concatenate([y, [-1,1]])
    n = len(y)
    pos = np.sum(y[y==1],dtype=float)/n
    neg = 1-pos
    return pos,neg


def naivebayesPXY(x,y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (dx1)
    negprob: probability vector of p(x|y=-1) (dx1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = x.shape
    x = np.concatenate([x, np.ones((2,d))])
    y = np.concatenate([y, [-1,1]])
    n, d = x.shape
    
    x1 = x[y==1]
    y1 = y[y==1]
    posprob=np.sum(x1,axis=0)/np.sum(x1)
    #dn = sum(np.inner(x1.T,y1))
    #posprob = sum(x1)/dn
    
    x_1 = x[y==-1]
    y_1 = y[y==-1]
    negprob = np.sum(x_1,axis=0)/np.sum(x_1)
    #dn_1 = sum(np.inner(x_1.T,y1))
    #negprob = -sum(x_1)/dn_1
    
    return posprob,negprob

def naivebayes(x,y,xtest):
    """
    function logratio = naivebayes(x,y);
    
    Computation of log P(Y|X=x1) using Bayes Rule
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    xtest: input vector of d dimensions (1xd)
    
    Output:
    logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
    """
    
    posprob,negprob = naivebayesPXY(x,y)
    pos,neg = naivebayesPY(x,y)
    logratio = (np.log(pos)+np.inner(xtest,np.log(posprob)))-(np.log(neg)+np.inner(xtest,np.log(negprob)))
    #logratio = (np.log(pos)+np.inner(xtest,np.log(posprob)))/(np.log(neg)+np.inner(xtest,np.log(negprob)))
    return logratio

def naivebayesCL(x,y):
    """
    function [w,b]=naivebayesCL(x,y);
    Implementation of a Naive Bayes classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)

    Output:
    w : weight vector of d dimensions
    b : bias (scalar)
    """
    
    n, d = x.shape
    pos,neg = naivebayesPY(x,y)
    b = np.log(pos/neg)
    posprob,negprob = naivebayesPXY(x,y)
    w = np.log(posprob/negprob)
    return w,b

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b);
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector of d dimensions
    b : bias (optional)
    
    Output:
    preds: predictions
    """
    
    w = w.reshape(-1)
    preds = np.array(np.sign(np.inner(w,x)+b),dtype=int)
    
    return preds

if __name__ == '__main__':
    dims = 5243 # you can choose any suitable number. 
    X,Y = genTrainFeatures("../Dataset/girls.train","../Dataset/boys.train",dimension=5243, fix=3)
    n,d = X.shape
    idx = np.random.permutation(n)
    xTr = X[idx,:][:int(0.8*n)]
    xTe = X[idx,:][int(0.8*n):]
    yTr = Y[idx][:int(0.8*n)]
    yTe = Y[idx][int(0.8*n):]
    print("first let's predict directly using the probabilities\n")
    logratio = naivebayes(xTr,yTr,xTe)
    preds = np.array(logratio>0)*2-1
    error = np.mean(preds != yTe)
    print('Test error: %.2f%%' % (100 * error))     
    print("\n")           
    print("Now, Naive Bayes can be used as linear classifier, i.e, we can find the weight vector (w) and  bias (b) for a decision boundary, we can recove w and b using our function naivebayesCL(x,y)")
    print("\n")   
    print("press enter to continue>")   
    inp = input()
    if len(inp) <1:
        w,b=naivebayesCL(xTr,yTr)
        error = np.mean(classifyLinear(xTe,w,b) != yTe)
        print('Training error: %.2f%%' % (100 * error))
    
    print("\n")   
    print("Now try some names yourself")
    print("\n")    
    while True:
        print('Please enter your name>')
        yourname = input()
        if len(yourname) < 1:
            break
        xtest = name2features(yourname,B=dims,LoadFile=False)
        pred = classifyLinear(xtest,w,b)[0]
        if pred > 0:
            print("%s, I am sure you are a nice boy.\n" % yourname)
        else:
            print("%s, I am sure you are a nice girl.\n" % yourname)
        
