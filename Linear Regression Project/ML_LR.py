import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
%matplotlib inline

# load the dataset
x_train, y_train = load_data("data/restaurant_data.txt")

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    """
    m = x.shape[0] 
    total_cost = 0
    for i in range(m):
        total_cost = total_cost + (w*x[i] + b - y[i])**2
    
    total_cost = total_cost/(2*m)

    return total_cost


# Public tests
from public_tests import *
compute_cost_test(compute_cost)

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 

    """
    
    m = x.shape[0]  
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        dj_dw = dj_dw + (w*x[i] + b - y[i])*x[i]
        dj_db = dj_db + w*x[i] + b - y[i]

    dj_dw = dj_dw/m
    dj_db = dj_db/m
         
    return dj_dw, dj_db



def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    """
    
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history 

# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
