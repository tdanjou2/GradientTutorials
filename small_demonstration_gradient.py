# initialization
import sympy # in case we need to derive
x = sympy.Symbol('x')

import matplotlib.pyplot as plt
import numpy as np

x0 = 10 # where we start the descent
learning_rate = 0.001 
precision = 0.0001
precision_parameter = 1 
max_iters = 50000 
cmpt = 1 # to make the while loop stop eventually - sanity check

# the function we want to minimize
def f(x):
    return -5*x**3 + 9*x

# the gradient of the function
def df(x) : 
    return -15*x**2 + 9 

# we store the values obtained during the process
X_list = [x0]
Y_list = [f(x0)]


while precision_parameter > precision and  cmpt < max_iters:
    x1 = x0 - abs(learning_rate * df(x0))
    precision_parameter = abs(x0-x1)
    x0 = x1
    X_list.append(x0)
    Y_list.append(f(x0))
    cmpt = cmpt+1 
    
print("f is minimum when x equals : ", x0)

# visualization of the function and its gradient

X = np.linspace(-5,5,500)
Y = [f(X[i])for i in range (len(X))]
plt.scatter(X_list,Y_list,c="g")
plt.plot(X_list,Y_list,c="g")
plt.plot(X,Y, c="r")
plt.title("Gradient descent")
plt.show()