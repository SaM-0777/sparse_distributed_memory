import torch
import torch.nn as nn


class SparseDistributedMemory(nn.Module):
    def __init__(
        self,
        n_dim: int,
        u_dim: int,
        m_dim: int,
        activation_radius_h: int,
        counter_range: int = 15,
    ) -> None:
        super(SparseDistributedMemory, self).__init__()

        self.N = n_dim  # Input dim (Length of address)
        self.U = u_dim  # Output dim (Length of word)
        self.M = m_dim  # Hidden dim (Number of hard locations)
        self.H = activation_radius_h
        self.D = n_dim - 2 * activation_radius_h  # Similarity threshold
        self.counter_range = counter_range

        # Address
        self.A = nn.Parameter(
            torch.randint(0, 2, (m_dim, n_dim), dtype=torch.float32) * 2 - 1,
            requires_grad=False,
        )

        # Content
        self.C = nn.Parameter(
            torch.zeros(m_dim, u_dim, dtype=torch.float32), requires_grad=False
        )

    def _hidden_layer_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the Hidden Neurons (Activation Vector y).
        d = A * x, then y_m = 1 if d_m >= D else 0
        """
        d = torch.matmul(self.A, x)  # [M, 1]

        y = (d >= self.D).float()
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieval
        z = sign(C^T * y)
        """
        y = self._hidden_layer_activations(x=x)

        # sums the active counters
        # s = C^T * y
        s = torch.matmul(self.C.T, y)  # [U, 1]

        # z_u = 1 if s_u > 0, else -1
        z = torch.where(s > 0, torch.tensor(1.0), torch.tensor(-1.0))
        return z

    def write(self, x: torch.Tensor, w: torch.Tensor) -> None:
        """
        Store word w (Hebbian Learning)
        C := C + y * w^T
        """
        y = self._hidden_layer_activations(x=x)
        updated_matrix = torch.outer(y, w)

        self.C.data += updated_matrix

        self.C.data.clamp_(-self.counter_range, self.counter_range)

    def network_stats(self, target_fidelity: float = 0.999):
        """
        Calculates network capacity statistics
        """
        if target_fidelity == 0.999:
            rho_required = 3.09
        elif target_fidelity == 0.99:
            rho_required = 2.33
        elif target_fidelity == 0.95:
            rho_required = 1.64
        else:
            rho_required = 3.09

        tau_max = 1 / (rho_required**2)
        T_max = int(tau_max * self.M)

        practical_max = int(0.05 * self.M)  # 5% of M

        stats = {
            "memory_size": self.M,
            "target_fidelity": target_fidelity,
            "theoretical_max_capacity": f"{tau_max:.3f} ({tau_max*100:.1f}%)",
            "theoretical_max_words (T_max)": T_max,
            "practical_max_words": practical_max,
        }

        for key, val in stats.items():
            print(f"{key:.<30} {val}")

        return stats
