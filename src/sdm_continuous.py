import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousDM(nn.Module):
    def __init__(self, n_dim, u_dim, m_dim, sigma=1.0):
        super(ContinuousDM, self).__init__()
        
        self.N = n_dim
        self.U = u_dim
        self.M = m_dim
        self.sigma = sigma
        
        # Address Matrix A
        self.A = nn.Parameter(F.normalize(torch.randn(m_dim, n_dim), p=2, dim=1), requires_grad=False)
        
        # Contents Matrix
        self.C = nn.Parameter(torch.zeros(m_dim, u_dim), requires_grad=False)
        
        self.stored_count = 0

    def _continuous_activation(self, x):
        """
        Computes continuous activation using Gaussian RBF kernel.
        y_m = exp(-||x - A_m||^2 / (2 * sigma^2))
        """
        dist_sq = 2.0 - 2.0 * torch.matmul(self.A, x) 

        y = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return y

    def forward(self, x):
        """Retrieval: s = C^T * y, then normalize."""
        y = self._continuous_activation(x)
        s = torch.matmul(self.C.T, y)

        z = F.normalize(s, p=2, dim=0)
        return z

    def write(self, x, w):
        """Storage: C := C + y * w^T"""
        y = self._continuous_activation(x)

        update_matrix = torch.outer(y, w)
        self.C.data += update_matrix
        
        self.stored_count += 1
