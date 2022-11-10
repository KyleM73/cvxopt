import torch
import cvxopt


A = torch.tensor([[2,1],[0,1]])
b = torch.tensor([[2],[1]])
w = torch.tensor([[1,0],[0,1]])
rho = torch.tensor(100)

LS = cvxopt.LSsolver(A,b,w,rho)
sol = LS.solve()
print(sol)
