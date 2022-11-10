import torch

class LSsolver:
    def __init__(self,A,b,w=None,rho=None):
        assert torch.is_tensor(A)
        assert torch.is_tensor(b)
        self.A = A.to(torch.float64)
        self.b = b.to(torch.float64)
        if w is not None:
            assert torch.is_tensor(w)
            assert w.size() == A.size()
            self.w = w.to(torch.float64)
        else:
            self.w = torch.eye(*A.size()).to(torch.float64)
        if rho is not None:
            assert torch.is_tensor(rho)
            if rho.size() == A.size():
                self.rho = rho.to(torch.float64)
            elif not [*torch.squeeze(rho).size()]:
                self.rho = torch.squeeze(rho)*torch.eye(*A.size()).to(torch.float64)
        else:
            self.rho = torch.zeros_like(A).to(torch.float64)

    def solve(self):
        inv_ = self.A.T @ self.w @ self.A + self.rho
        try:
            inv = torch.linalg.solve(inv_,torch.eye(*inv_.size()).to(torch.float64))
        except RuntimeError:
            print("Warning: matrix not invertable.")
            return
        return inv @ self.w @ self.A.T @ self.b
