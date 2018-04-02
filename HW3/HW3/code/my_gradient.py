import numpy as np
import math
from numpy.linalg import inv
import sys
np.random.seed(789)

def rosenbrock(x, a, b):
    '''
    Compute and return the value of the Rosenbrock function at input x
    :param x: numpy vector of shape (2,)
    :param a: scalar for the variable a in the Rosenbrock function
    :param b: scalar for the variable b in the Rosenbrock function
    :return: scalar
    '''
    ##f(x) = (a-x0)^2 + b(x1-x0^2)^2
    f = math.pow(a-x[0],2)  ### first term (a-x0)^2
    f = f + b * math.pow((x[1] - math.pow(x[0],2)), 2)

    return f

def rosenbrock_grad(x, a, b):
    '''
    Compute the gradient of the Rosenbrock function at point x
    :param x: numpy vector of shape (2,)
    :param a: scalar for the variable a in the Rosenbrock function
    :param b: scalar for the variable b in the Rosenbrock function
    :return: ndarray vector of shape (2,)
    '''
    ### z = [ a - x0, x1 - x0]
    ### Df = [ 4b*x0*(x1-x0^2)^2 - 2(a-x0), 2*x1(x1-x0^2) ]
    df = np.zeros(shape=[9,9])

    test_1 = 4*b*x[0] - 4*b*x[0]*x[1]+2*x[0] -2
    test_2 = 2*b*x[1] -2*b*math.pow(x[0],2)
    df[0] = test_1
    df[1] = test_2
    #print (df, gf, test_1, test_2 )

    return df

def rosenbrock_hessian(x, a, b):
    '''
    Compute the hessian of the Rosenbrock function at point x
    :param x: numpy vector of shape (2,)
    :param a: scalar for the variable a in the Rosenbrock function
    :param b: scalar for the variable b in the Rosenbrock function
    :return: ndarray vector of shape (2,2)
    '''
    ###[-4b*(x1-x0^2)+8bx^2 +2  -4bx]
    ###[-4bx                     2b ]
    hes = np.array([[0, 0], [0, 0]], dtype = float)        # return this 2d array
    hes[0,0] = -4*b*(x[1] - math.pow(x[0],2)) + 8*b*math.pow(x[0],2) + 2
    hes[0,1] = -4*b*x[0]
    hes[1,0] = -4*b*x[0]
    hes[1,1] = 2*b



    return hes

def gradient_descent(fn, grad_fn, x0, lr, threshold=1e-10, max_steps=100000):
    '''
    compute the gradient descent of a function until the minimum threshold is obtained or max_steps is reached
    :param fn: function that takes vector of size x0 as input and outputs a scalar
    :param grad_fn: gradient of the fn function
    :param x0: the initial starting location of x0
    :param lr: the step size
    :param threshold: minimum threshold to check for convergence of the function output
    :param max_steps: maximum number of steps to take
    :return: tuple of (scalar: ending fn value, ndarray: final x, int: number of iterations)
    '''

    fx = 0
    step = 0
    temp = 0

    """
    print(fn(x0))
    print(x0)
    print(grad_fn(x0))
    print("test var above")
    """
    xk = x0         #current x
    while (step < max_steps) and (temp < max_steps):
        step = step + 1
        temp = temp + 1
        fx = fn(xk)
        #print(xk, grad_fn(xk))
        xk = xk - lr*(grad_fn(xk))        #updating xk+1
        diff = fn(xk) - fx          #diff of two functions
        diff = abs(diff)
        fx = fn(xk)
        #print( diff)

        if diff < threshold:                #if converge, then break & return
            temp = max_steps
            #print("hit!!!!!!!!!")


    #print(x0, "ended with step: ", step, "and diff: ", diff)


    return fx, xk, step


def newton_method(fn, grad_fn, hessian_fn, x0, lr, threshold=1e-10, max_steps=100000):
    '''
    find the parameters that minimize a function fn using Newton's Method.
    To invert the hessian use the function np.linalg.
    :param fn: function that takes vector of size x0 as input and outputs a scalar
    :param grad_fn: gradient of the fn function
    :param hessian_fn:
    :param x0: the initial starting location of x0
    :param lr: the step size
    :param threshold: minimum threshold to check for convergence of the function output
    :param max_steps: maximum number of steps to take
    :return: tuple of (scalar: ending fn value, ndarray: final x, int: number of iterations)
    '''
    xk = x0
    fx = fn(x)
    step = 0
    temp = 0
    count = 0
    last = 1
    while (step < max_steps) and (temp < max_steps):
        step = step + 1
        temp = temp + 1
        count = count + 1

        fx = fn(xk)
        hs = inv(hessian_fn(xk))
        step_dir = np.matmul(hs,grad_fn(xk))        # hes*grad
        #print(step_dir*lr)
        xk = xk - lr*step_dir       #updating xk+1

        diff = fn(xk) - fx          #diff of two functions, new - old

        #print(xk, diff, step_dir, tiral)
        fx = fn(xk)

        if abs(diff) < threshold:                #if converge, then break & return
            temp = max_steps
            print("hit!!!!!!!!!")
        if count == 10:
            #print(xk,diff,step_dir)
            count = 0
        if abs(last) < abs(diff):
            print( hs, fx)
        if abs(diff)>1e+10:
            sys.exit()



        last = diff

    print(x0, "ended with step: ", step, "and diff: ", diff)


    return fx, xk, step



if __name__ == '__main__':
    # these lambda functions define the Rosenbrock functions with fixed 'a' and 'b' values
    rosen_fn = lambda x: rosenbrock(x, a=1, b=100)
    rosen_grad_fn = lambda x: rosenbrock_grad(x, a=1, b=100)
    rosen_hess_fn = lambda x: rosenbrock_hessian(x, a=1, b=100)

    # This runs gradient descent on on the Rosenbrock function and prints out the result
    x0 = np.zeros(2)
    val, x, itrs = gradient_descent(rosen_fn, rosen_grad_fn,  x0=x0, lr=0.001)

    print(val, itrs, x)


    # This runs Newton's method on the Rosebrock function
    x0 = np.zeros(2)
    val, x, itrs = newton_method(rosen_fn, rosen_grad_fn, rosen_hess_fn, x0=x0, lr=1.0)
    print(val, itrs, x )


    # run gradient descent from random points on [(-1, -1), (1, 1)] with different learning rates
    num_trials = 10
    lrs = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.0025]
    results = []
    for lr in lrs:
        lr_res = []
        print('Running trials of gradient descent with lr {0}'.format(lr))
        for trial in range(num_trials):
            x0 = np.random.random(2) * 2 -1
            val, x, itrs = gradient_descent(rosen_fn, rosen_grad_fn,  x0=x0, lr=lr)
            lr_res.append([np.linalg.norm(x-np.ones_like(x)), itrs])
        lr_mn = np.array(lr_res).mean(axis=0)
        lr_std = np.array(lr_res).std(axis=0)
        results.append([lr, lr_mn[0], lr_std[0], lr_mn[1], lr_std[1]])

    print("\n")
    print("Learning rate | mean solution error +- std | mean iterations +- std")
    for i in range(len(results)):
        print("{0:.4f}\t{1:.5f}+-{2:.5f}\t{3:>#08.1f}+-{4:.2f}".format(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]))

    # run newton's method from random points on [(-1, -1), (1, 1)] with different learning rates
    num_trials = 10
    lrs = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = []
    for lr in lrs:
        lr_res = []
        print("Running trials of Newtown's Method with lr {0}".format(lr))
        for trial in range(num_trials):
            x0 = np.random.random(2) * 2 - 1
            val, x, itrs = newton_method(rosen_fn, rosen_grad_fn, rosen_hess_fn, x0=x0, lr=lr)
            lr_res.append([np.linalg.norm(x - np.ones_like(x)), itrs])
        lr_mn = np.array(lr_res).mean(axis=0)
        lr_std = np.array(lr_res).std(axis=0)
        results.append([lr, lr_mn[0], lr_std[0], lr_mn[1], lr_std[1]])

    print("\n")
    print("Learning rate | mean solution error +- std | mean iterations +- std")
    for i in range(len(results)):
        print("{0:.4f}\t{1:.5f}+-{2:.5f}\t{3:>#08.1f}+-{4:.2f}".format(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]))
    """
"""
