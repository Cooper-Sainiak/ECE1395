import numpy as np
from computeCost import computeCost

# Q1 Part B
def gradientDescent(X,y,theta_init,alpha,num_iters):
    m = len(y)
    theta = theta_init.copy()
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        prediction = X.dot(theta)
        error = prediction - y
        theta = theta - (alpha/m)*X.T.dot(error)
        cost_history[i] = computeCost(X,y,theta)

    return theta, cost_history