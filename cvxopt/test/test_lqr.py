import torch
import time
import cvxopt

## double integrator of dim n
n = 2

## continuous state space
A = torch.zeros(2*n,2*n)
A[:n,n:] = torch.eye(n)
B = torch.zeros(2*n,n)
B[n:,:] = torch.eye(n)

## discrete state space
T = 0.01 #sampling time
Ad = torch.eye(2*n) + A * T
Bd = B * T

## lqr params
Q = 1
R = 1
N = 0
H = 10
dt = T

x_initial = torch.rand(2*n,1)
now = time.time()
LQR = cvxopt.LQRsolver(Ad,Bd,Q,R,N,H,dt)
print('computed LQR gains in {} seconds'.format(time.time()-now))
K,u,x = LQR.solve(x_initial)
print('total computation time: {} seconds\n'.format(time.time()-now))
print('x_initial: {}'.format(x_initial))
print('x_final: {}'.format(x[-1,...]))
print('control effort: {}'.format(torch.sum(torch.abs(u))))
