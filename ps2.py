import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn
import pandas as pd
import time

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
plt.title('Cost vs. Iteration number')
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

print("Mileage Mean:", feat_mean[0])
print("Mileage Std:", feat_std[0])
print("MPG Mean:", feat_mean[1])
print("MPG Std:", feat_std[1])

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
y_train = y[train]
y_test = y[test]

print("X training set size:", len(X_train))
print("X testing set size:", len(X_test))
print("y training set size:", len(y_train))
print("y testing set size:", len(y_test))

# Q2 Part D
# 1
np.random.seed(32)
df = pd.read_csv('ford.csv')
feat = 'mileage'
output = 'price'

X_init = df[[feat]].values
y = df[output].values
feat_mean = np.mean(X_init)
feat_std = np.std(X_init)

X_standard = (X_init - feat_mean) / feat_std

m = len(y)
X = np.column_stack([np.ones(m),X_standard])

index = np.random.permutation(m)
split = int(0.9 * m)
train = index[:split]
test = index[split:]

X_train = X[train]
X_test = X[test]
y_train = y[train]
y_test = y[test]

alpha = 0.01 # same value I used before
num_iters = 500 + (32 * 5)
theta_init = np.array([0.0, 0.0])

theta, cost_history = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)
print("Theta_0:", theta[0])
print("Theta_1:", theta[1])
# 2
# cost vs. iteration number
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history, linewidth=2)
plt.xlabel('Iteration number')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration number')
plt.grid(True)
plt.show()

# regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_init[train], y_train, alpha=0.5, color='blue', label='Training data')
x_range = np.linspace(X_init.min(), X_init.max(), 100)
x_range_standard = (x_range - feat_mean) / feat_std
x_bias = np.column_stack([np.ones(len(x_range)), x_range_standard])
y_pred = x_bias.dot(theta)

plt.plot(x_range, y_pred, 'r-', linewidth=2, label='Regression line')
plt.xlabel('Mileage - mi')
plt.ylabel('Price - $')
plt.title('Univariate Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Q2 Part E
np.random.seed(32)
df = pd.read_csv('ford.csv')
feat = ['mileage','mpg']
output = 'price'

X_init = df[feat].values
y = df[output].values
feat_mean = np.mean(X_init,axis=0)
feat_std = np.std(X_init,axis=0)
X_standard = (X_init - feat_mean) / feat_std

m = len(y)
X = np.column_stack([np.ones(m),X_standard])

index = np.random.permutation(m)
split = int(0.9 * m)
train = index[:split]
test = index[split:]

X_train = X[train]
X_test = X[test]
y_train = y[train]
y_test = y[test]

# Gradient descent
alpha = 0.01
num_iters = 750 + (32 * 5)
theta_init = np.zeros(3)
start_time = time.time()
thetaGD, cost_history = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)
GDtime = time.time() - start_time
print('GD theta:', thetaGD)
print('GD time:', GDtime)

# Normal Equation
start_time = time.time()
thetaNorm = normalEqn(X_train,y_train)
Ntime = time.time() - start_time
print('Normal theta:', thetaNorm)
print('Normal time:', Ntime)

# cost vs. iteration number GD
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history, linewidth=2)
plt.xlabel('Iteration number')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration number')
plt.grid(True)
plt.show()

# Q2 Part F
mseGD = computeCost(X_test,y_test,thetaGD)
mseNorm = computeCost(X_test,y_test,thetaNorm)

df = pd.read_csv('ford.csv')
feat = 'mileage'
output = 'price'

X_init_uni = df[[feat]].values
y_uni = df[output].values
feat_mean_uni = np.mean(X_init_uni)
feat_std_uni = np.std(X_init_uni)

X_standard_uni = (X_init_uni - feat_mean_uni) / feat_std_uni

m = len(y_uni)
X_uni = np.column_stack([np.ones(m),X_standard_uni])

np.random.seed(32)
index = np.random.permutation(m)
split = int(0.9 * m)
train_uni = index[:split]
test_uni = index[split:]

X_train = X_uni[train_uni]
X_test = X_uni[test_uni]
y_train = y_uni[train_uni]
y_test = y_uni[test_uni]

alpha = 0.01 # same value I used before
num_iters = 750 + (32 * 5)
theta_init = np.zeros(2)

theta_uni, cost_history_uni = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)

mseUni = computeCost(X_test,y_test,theta_uni)

print('Univariate Model (mileage only):', mseUni)
print('Multivariate Gradient Descent:', mseGD)
print('Multivariate Normal Equation:', mseNorm)

# Q2 Part G
np.random.seed(32)
df = pd.read_csv('ford.csv')
feat = ['mileage','mpg']
output = 'price'

X_init = df[feat].values
y = df[output].values
feat_mean = np.mean(X_init,axis=0)
feat_std = np.std(X_init,axis=0)
X_standard = (X_init - feat_mean) / feat_std

m = len(y)
X = np.column_stack([np.ones(m),X_standard])

index = np.random.permutation(m)
split = int(0.9 * m)
train = index[:split]
test = index[split:]

X_train = X[train]
X_test = X[test]
y_train = y[train]
y_test = y[test]

num_iters = 300
theta_init = np.zeros(3)

# too small
alpha = 0.001
theta1, cost_history1 = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history1, linewidth=2)
plt.xlabel('Iteration Number')
plt.ylabel('Cost')
plt.title('Learning Rate 0.001')
plt.grid(True, alpha=0.3)
plt.show()
# appropriate
alpha = 0.01
theta2, cost_history2 = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history2, linewidth=2)
plt.xlabel('Iteration Number')
plt.ylabel('Cost')
plt.title('Learning Rate 0.01')
plt.grid(True, alpha=0.3)
plt.show()
# oscillating but converging
alpha = 1.79
theta3, cost_history3 = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history3, linewidth=2)
plt.xlabel('Iteration Number')
plt.ylabel('Cost')
plt.title('Learning Rate 1.79')
plt.grid(True, alpha=0.3)
plt.show()
# diverging
alpha = 2.0
theta4, cost_history4 = gradientDescent(X_train,y_train,theta_init,alpha,num_iters)
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history4, linewidth=2)
plt.xlabel('Iteration Number')
plt.ylabel('Cost')
plt.title('Learning Rate 2.0')
plt.grid(True, alpha=0.3)
plt.show()