import numpy as np

# Q1 Part A
def computeCost(X,y,theta):
    m = len(y)
    prediction = X.dot(theta)
    error = prediction - y
    J = (1/(2*m))*np.sum(error**2)

    return J