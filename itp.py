## Implementation of the interior point method for Linear programming problem
## author Syed Anjum Haider
## Problem statement : min f(x) s.t Ax = b and x>=0
import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

def interior_point_method(A,B,C,x,xu,eps):
    cond = 3
    while cond > 2:
        x1d = x.reshape(-1)
        D =  np.diag(x1d)
        AT = np.transpose(A)
        CT = np.transpose(C)
        BT = np.transpose(B)
        XT = np.transpose(x)
        ADT = multi_dot([A,D,D,AT])
        ADT_inv = np.linalg.inv(ADT)
        w_k = multi_dot([ADT_inv,A,D,D,CT])
        r_k = CT-AT.dot(w_k)
        error = multi_dot([xu,D,r_k])
        if (all(i >= 0 for i in r_k) and error<=eps) :
            cond = 1
        Y = multi_dot([D,D,r_k])
        Z = D.dot(r_k)
        N = np.linalg.norm(Z)
        XT = XT - 0.59*Y/N
        x = np.transpose(XT)

    return x

A = pd.read_csv("a.csv")
B = pd.read_csv("b.csv")
C = pd.read_csv("c.csv")
##initial guess for the time being taken from input script
xo = pd.read_csv("x.csv")
xu = pd.read_csv("xu.csv")

A = np.array(A)
B = np.array(B)
C = np.array(C)
xo = np.array(xo)
xu = np.array(xu)
eps = 1e-10
print("shape of the constraint==",A.shape)
result = interior_point_method(A,B,C,xo,xu,eps)
print("The solution of the optimization problem==",result)
