import torch

from src.sdm import SparseDistributedMemory


N, U, M = 256, 256, 1000
H = 110

sdm = SparseDistributedMemory(n_dim=N, u_dim=U, m_dim=M, activation_radius_h=H, counter_range=15)

for i in range(1, 120): # try to store 119 words to overfit
    address = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1
    word = torch.randint(0, 2, (U,), dtype=torch.float32) * 2 - 1
    
    sdm.write(address, word)
    
    if i in [1, 25, 50, 100, 119]:
        sdm.network_stats(target_fidelity=0.999)
        
        if i == 1:
            first_address = address.clone()
            first_word = word.clone()
            
            retrieved = sdm(first_address)
            accuracy = (retrieved == first_word).float().mean().item() * 100
            print(f"-> Retrieval accuracy of Word #1: {accuracy:.2f}%\n")

