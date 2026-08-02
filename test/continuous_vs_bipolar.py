import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.sdm import SparseDistributedMemory
from continuous_dm import ContinuousDM

# --- 1. Setup Parameters ---
N, U, M = 256, 256, 1000
H = 110          # Radius for Bipolar
sigma = 0.2      # Bandwidth for Continuous (tuned for unit vectors)

bipolar_sdm = SparseDistributedMemory(n_dim=N, u_dim=U, m_dim=M, activation_radius_h=H)
cont_sdm = ContinuousDM(n_dim=N, u_dim=U, m_dim=M, sigma=sigma)

# --- 2. Generate Data ---
T = 50 # Number of words to store (5% of M)

# Continuous ground truth (unit vectors)
cont_addresses = F.normalize(torch.randn(T, N), p=2, dim=1)
cont_words = F.normalize(torch.randn(T, U), p=2, dim=1)

# Bipolar ground truth (binarized)
bip_addresses = torch.sign(cont_addresses)
bip_words = torch.sign(cont_words)

# --- 3. Store Data ---
for i in range(T):
    bipolar_sdm.write(bip_addresses[i], bip_words[i])
    cont_sdm.write(cont_addresses[i], cont_words[i])

# --- 4. Benchmark Noise Tolerance ---
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # Standard deviation of Gaussian noise
bipolar_accuracies = []
cont_accuracies = []

target_idx = 0 # Let's try to retrieve the first word we stored

for noise in noise_levels:
    # Create noisy address
    noisy_cont = cont_addresses[target_idx] + torch.randn(N) * noise
    noisy_cont = F.normalize(noisy_cont, p=2, dim=0) # Re-normalize
    
    # Bipolar noisy address (flip bits based on noise magnitude)
    noisy_bip = bip_addresses[target_idx].clone()
    flip_prob = 0.5 * (1 - math.exp(-noise)) # Map continuous noise to bit flip probability
    flip_mask = torch.rand(N) < flip_prob
    noisy_bip[flip_mask] *= -1

    # --- Retrieve ---
    retrieved_bip = bipolar_sdm(noisy_bip)
    retrieved_cont = cont_sdm(noisy_cont)

    # --- Evaluate ---
    # Bipolar: % of matching bits
    bip_acc = (retrieved_bip == bip_words[target_idx]).float().mean().item() * 100
    
    # Continuous: Cosine Similarity (1.0 is perfect, 0.0 is random)
    cont_acc = F.cosine_similarity(retrieved_cont.unsqueeze(0), cont_words[target_idx].unsqueeze(0)).item() * 100

    bipolar_accuracies.append(bip_acc)
    cont_accuracies.append(cont_acc)

# --- 5. Print Results ---
print(f"{'Noise Level':<15} | {'Bipolar SDM (% Match)':<25} | {'Continuous SDM (% Cosine Sim)':<30}")
print("-" * 75)
for i, noise in enumerate(noise_levels):
    print(f"{noise:<15.2f} | {bipolar_accuracies[i]:<25.2f} | {cont_accuracies[i]:<30.2f}")
