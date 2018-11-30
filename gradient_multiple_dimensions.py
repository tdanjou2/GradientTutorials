import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

lr = 0.0001 # learning rate
max =  10000 # maximum iterations
p = 1 # number of parameters
theta = np.array([[1] for i in range (p+1)]) # initialization of theta
n = 100 # number of observations
precision = 0.1 # precision

def creation_X(): # creating the parameters
    X0 = np.array([[1] for i in range (n)])
    for k in range (p):
        X0 = np.c_[X0,np.array(sorted(100 * np.random.rand(n,1)))]
    X = X0
    return X
X = creation_X()

# creating the results
y = np.array([ [X[i][0] + X[i][1] + 5*(-1)**(i%2)*np.random.rand()] for i in range (n)])

def cost_function (theta,X,y,n):
    cost = (1/(2*n))*np.sum(np.square(X.dot(theta)-y))
    return cost

def gradient_decent (theta=theta):
    compt = 0
    Xt = X.transpose()
    while (compt < max and cost_function(theta,X,y,n) > precision ):
        theta = theta - (1/n)*lr*np.dot(Xt,(np.dot(X,theta)-y))
        compt += 1
        if compt%10 == 0 :
            print(cost_function(theta,X,y,n), theta)

    return theta
theta = gradient_decent(theta)
# recuperation of X1
X1 = []
b = []
for i in range (n):
    X1.append(float(X[i][1]))
    b.append(theta[0]+float(X[i][1])*theta[1])

plt.plot(X1,y,'r')
plt.plot(b,y,'b')
plt.show()