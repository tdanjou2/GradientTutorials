import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from numpy.linalg import inv

max =  10000 # maximum iterations
p = 1 # number of parameters
n = 100 # number of observations

def creation_X(): # creating the parameters
    X0 = np.array([[1] for i in range (n)])
    for k in range (p):
        X0 = np.c_[X0,np.array(sorted(100 * np.random.rand(n,1)))]
    X = X0
    return X

X = creation_X()

# creating the results
y = np.array([ [X[i][0] + X[i][1] + 5*(-1)**(i%2)*np.random.rand()] for i in range (n)])

Xt = X.transpose()
theta = np.dot(np.dot(inv(np.dot(Xt,X)),Xt),y)

# recuperation of X1
X1 = []
for i in range (n):
    X1.append(float(X[i][1]))

plt.plot(X1,y,'r')
plt.plot(X1,np.dot(X,theta),'b')
plt.show()