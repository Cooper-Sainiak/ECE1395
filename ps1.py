import numpy as np
import matplotlib.pyplot as plt
import time

#D = 32
#Q2a
mu = 2 + (32/100)
sigma = 0.5 + (32/200)
n = 1000000

x = sigma*np.random.randn(n, 1) + mu

#Q2b
minimum = -32/50
maximum = 32/100

z = np.random.uniform(minimum, maximum, size=(n,1))

#Q2c
#x histogram
plt.figure()
plt.hist(x.flatten(), bins=100, density=True)
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("x-Gaussian")
plt.show()

#z histogram
plt.figure()
plt.hist(z.flatten(), bins=100, density=True)
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("z-Uniform")
plt.show()

#Q2d
#1.
x_2 = x.copy()
start = time.time()
for i in range(x_2.shape[0]):
    x_2[i,0] += 2
end = time.time()
print("1. Loop time =", end-start, "seconds")

#2.
x_3 = x.copy()
start = time.time()
x_3 = x_3 + 2
end = time.time()
print("2. Vectorized Ops time =", end-start, "seconds")

#Q2e
z_2 = z.copy()
y = z[(z>0) & (z<0.8)]
count = y.size
print("y size:", count)

#Q3a
A = np.array([[2,10,8],
              [3,5,2],
              [6,4,4]])

print("Min in each column:", A.min(axis=0))
print("Max in each row:", A.max(axis=1))
print("Smallest in entire matrix:", A.min())
row_sum = A.sum(axis=1)
print("Sum of rows:", row_sum)
print("Sum of all elements:", A.sum())
B = A**2
print("Matrix B:\n", B)

print("Verification line:", B[0,1]==(A[0,1]**2))

#Q3b
threeB = np.array([[2,5,-2],
                   [2,6,4],
                   [6,8,18]])
rightSide = np.array([32,6,15])
answer = np.linalg.solve(threeB, rightSide)
x,y,z = answer
print("x =", x)
print("y =", y)
print("z =", z)

#Q4a
# the matrices will be renamed as X = fourX and y = fourY because x and y were used in the ps already
i = np.arange(1,11).reshape(10,1)
fourX = np.hstack((i, i**2, 32*i))
fourY = 3*i + 32

print("X: \n", fourX)
print("y: \n", fourY)

#Q4b
indices = np.random.permutation(10)
train = indices[:8]
test = indices[8:]

X_train = fourX[train, :]
X_test = fourX[test, :]
y_train = fourY[train, :]
y_test = fourY[test, :]

print("X_train: \n", X_train)
print("\nX_test: \n", X_test)
print("\ny_train: \n", y_train)
print("\ny_test: \n", y_test)

#Q4c
np.random.seed(0)
indices = np.random.permutation(10)
train = indices[:8]
test = indices[8:]

X_trainR = fourX[train, :]
X_testR = fourX[test, :]
y_trainR = fourY[train, :]
y_testR = fourY[test, :]

print("X_trainR: \n", X_trainR)
print("\nX_testR: \n", X_testR)
print("\ny_trainR: \n", y_trainR)
print("\ny_testR: \n", y_testR)