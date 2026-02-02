import numpy as np

def computeCost(X,y,theta):
    m = len(y)
    prediction = X.dot(theta)
    errors = prediction - y
    J = (1/(2*m))*np.sum(errors ** 2)

    return J