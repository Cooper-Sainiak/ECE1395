import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn
import pandas as pd

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

# Q2 Part B
# When testing my code, you might have to put ford.csv into the same folder to make the path work
df = pd.read_csv('ford.csv')
feat = ['mileage','mpg']
output = 'price'

# mileage vs. price
plt.figure(figsize=(8, 6))
plt.scatter(df['mileage'], df['price'], alpha=0.5, color='blue')
plt.xlabel('Mileage - mi')
plt.ylabel('Price - $)')
plt.title('Mileage vs Price')
plt.grid(True, alpha=0.3)
plt.show()

# mpg vs. price
plt.figure(figsize=(8, 6))
plt.scatter(df['mpg'], df['price'], alpha=0.5, color='red')
plt.xlabel('Fuel Usage - mpg')
plt.ylabel('Price - $)')
plt.title('MPG vs Price')
plt.grid(True, alpha=0.3)
plt.show()

# Q2 Part C
# to satisfy 5
np.random.seed(32)

# 2
X_init = df[feat].values
y = df[output].values
feat_mean = np.mean(X_init,axis=0)
feat_std = np.std(X_init,axis=0)
X_standard = (X_init - feat_mean) / feat_std
# 3
m = len(y)
X = np.column_stack([np.ones(m),X_standard])
# 4
index = np.random.permutation(m)
split = int(0.9 * m)
train = index[:split]
test = index[split:]

X_train = X[train]
X_test = X[test]
X_train = X[train]
X_test = X[test]