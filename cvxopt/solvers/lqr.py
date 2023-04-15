import torch

class LQRsolver(torch.nn.Module):
    def __init__(self,A,B,Q,R,N=0,H=10,dt=0.01):
        super(LQRsolver, self).__init__()
        self.A = A
        self.B = B
        if isinstance(Q,int) or isinstance(Q,float):
            self.Q = torch.eye(self.A.size()[1]) * Q
        else:
            self.Q = Q
        if isinstance(R,int) or isinstance(R,float):
            self.R = torch.eye(self.B.size()[1]) * R
        else:
            self.R = R
        if isinstance(N,int) or isinstance(N,float):
            self.N = torch.eye(self.A.size()[1],self.B.size()[1]) * N
        else:
            self.N = N
        self.H = H
        self.dt = dt
        self.steps = int(self.H/self.dt)

        self.P = torch.zeros(self.steps+1,*self.Q.size())
        self.P[self.steps,...] = self.Q

        self.K = torch.zeros(self.steps,self.B.size()[1],self.A.size()[1])

        for i in range(self.steps,0,-1):
            self.K[i-1,...] = torch.linalg.pinv(self.R + self.B.T @ \
                self.P[i,...] @ self.B) @ (self.B.T @ self.P[i,...] @ self.A + self.N.T)
            self.P[i-1,...] = self.A.T @ self.P[i,...] @ self.A - \
                (self.A.T @ self.P[i,...] @ self.B + self.N) @ \
                torch.linalg.pinv(self.R + self.B.T @ self.P[i,...] @ \
                self.B) @ (self.B.T @ self.P[i,...] @ self.A + self.N.T) + self.Q

    def forward(self,x):
        u = torch.zeros(self.steps,self.B.size()[1],1)
        x_star = torch.zeros(self.steps+1,self.A.size()[1],1)
        x_star[0,...] = x
        for i in range(self.steps):
            u[i,...] = -self.K[i,...] @ x_star[i,...]
            x_star[i+1,...] = self.A @ x_star[i,...] + self.B @ u[i,...]
        return self.K,u,x_star

    def solve(self,x):
        return self.forward(x)

    def get_K(self):
        return self.K



