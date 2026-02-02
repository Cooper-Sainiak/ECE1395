import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn

# D = 32

x1 = np.array([0,1,2,3])
x2 = np.array([1,1.5,4,2])
y = np.array([(1.5 + (32/100)),4,8.5,(8.5 + (32/50))])

# Q1 Part D Task 1
m = len(y)
X = np.column_stack([np.ones(m),x1,x2])

# Q1 Part D Task 2
test_theta = np.array([0.5,2,1])
cost = computeCost(X,y,test_theta)
print("Cost =", cost)

# Q1 Part D Task 3
theta_init = np.array([0,0,0])
alpha = 0.01
num_iters = 100 + (32*10)

thetaGD, cost_history = gradientDescent(X,y,theta_init,alpha,num_iters)
print("Theta:", thetaGD)

plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history, linewidth=2)
plt.xlabel('Iteration number')
plt.ylabel('Cost')
plt.title(f'Cost vs. Iteration number')
plt.grid(True)
plt.show()

# Q1 Part D Task 4
thetaNorm = normalEqn(X,y)
print("Theta:", thetaNorm)