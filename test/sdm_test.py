import torch

from src.sdm import SparseDistributedMemory


N = 256 # Address length
U = 256 # Word length
M = 1000 # Number of hard locations (Hidden neurons)
H = 110 # Radius of activation

sdm_net = SparseDistributedMemory(n_dim=N, u_dim=U, m_dim=M, activation_radius_h=H, counter_range=15)

original_address = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1
original_word = torch.randint(0, 2, (U,), dtype=torch.float32) * 2 - 1

print("Writing pattern to memory...")
sdm_net.write(original_address, original_word)

retrieved_exact = sdm_net(original_address)
exact_accuracy = (retrieved_exact == original_word).float().mean().item() * 100
print(f"Retrieval accuracy with EXACT address:   {exact_accuracy:.2f}%")


noise_percentage = 0.15
noisy_address = original_address.clone()
# flip 15% of the bits randomly
num_flips = int(N * noise_percentage)
flip_indices = torch.randperm(N)[:num_flips]
noisy_address[flip_indices] *= -1

retrieved_noisy = sdm_net(noisy_address)
noisy_accuracy = (retrieved_noisy == original_word).float().mean().item() * 100
print(f"Retrieval accuracy with NOISY address:   {noisy_accuracy:.2f}%")
