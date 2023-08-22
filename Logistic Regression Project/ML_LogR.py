import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

%matplotlib inline

# load dataset
X_train, y_train = load_data("data/test_admission_data.txt")

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

def sigmoid(z):
    """
    Compute the sigmoid of z

    """
    p = np.exp(-z)
    g = 1/(1+p)
    return g

def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples

    """

    m, n = X.shape
    total_cost = 0
    
    for i in range(m):
        z = np.dot(X[i],w) + b
        f_xi = sigmoid(z)
        loss = -y[i]*np.log(f_xi) - (1 - y[i])*np.log(1 - f_xi)
        total_cost = total_cost + loss
    
    total_cost = total_cost/m

    return total_cost

m, n = X_train.shape

def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 

    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
   
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    """
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    """
    m, n = X.shape   
    p = np.zeros(m)
   
    for i in range(m):   
        z_wb = np.dot(X[i],w) + b
        f_xi = sigmoid(z_wb)
        
        p[i] = 1 if f_xi>0.5 else 0
    
    return p

p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

X_train_new, y_train_new = load_data("data/microchip_data.txt")

print("Original shape of data:", X_train_new.shape)

mapped_X =  map_feature(X_train_new[:, 0], X_train_new[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples

    """

    m, n = X.shape
    
    cost_without_reg = compute_cost(X, y, w, b) 
    reg_cost = 0.
    reg_cost = sum(np.square(w))
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + (lambda_/(2 * m)) * reg_cost

    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization

    """
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
                      
    return dj_db, dj_dw

# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(mapped_X.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(mapped_X, y_train_new, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)

plot_decision_boundary(w, b, mapped_X, y_train_new)
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

#Compute accuracy on the training set
p = predict(mapped_X, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train_new) * 100))