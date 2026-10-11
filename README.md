# Sparse Distributed Memory

A PyTorch implementation of associative memory inspired by Pentti Kanerva’s sparse distributed memory (SDM). The repository contains two related models:

- *SparseDistributedMemory*: a bipolar, hard-location SDM for binary addresses and binary words.
- *ContinuousDM*: a continuous associative memory that uses Gaussian radial-basis activations for unit-normalized vectors.

The code is organized as small research/learning experiments. The scripts in test/ generate random data, write it to memory, retrieve it with exact or noisy addresses, and print accuracy or capacity statistics.

## Repository layout

| Path | Purpose |
| --- | --- |
| src/sdm.py | Bipolar Kanerva-style sparse distributed memory implementation. |
| src/continuous_dm.py | Continuous Gaussian-kernel distributed memory implementation. |
| main.py | Minimal entry point that prints a greeting. |
| test/sdm_test.py | Single-pattern exact and noisy-address retrieval example. |
| test/sdm_bulk_test.py | Incremental storage experiment with capacity statistics. |
| test/continuous_vs_bipolar.py | Noise comparison between the bipolar and continuous variants. |
| pyproject.toml | Project metadata and direct dependencies. |
| uv.lock | Locked dependency resolution for reproducible uv environments. |

## How the implementation works

### Bipolar sparse distributed memory

SparseDistributedMemory represents an address and a word as vectors in {−1, +1}. Its constructor accepts:

| Argument | Meaning |
| --- | --- |
| n_dim | Address length N. |
| u_dim | Word/output length U. |
| m_dim | Number of hard locations/hidden neurons M. |
| activation_radius_h | Hamming-radius threshold H. |
| counter_range | Symmetric clamp applied to content counters. |

The model initializes a fixed random address matrix A ∈ {−1,+1}^(M×N) and a zero content-counter matrix C ∈ ℝ^(M×U). For an input address x:

1. Each hard location computes d = A x.
2. A location activates when d_m ≥ D, where D = N − 2H.
3. Because bipolar dot products satisfy A_m · x = N − 2·Hamming(A_m, x), this threshold selects hard locations within Hamming distance H.
4. The activation vector y is binary, with one entry per hard location.

Writing a word w applies the distributed Hebbian-style update:

~~~text
C ← clamp(C + y wᵀ, −counter_range, +counter_range)
~~~

Reading computes the accumulated vote from active hard locations:

~~~text
s = Cᵀ y
z_u = +1 when s_u > 0, otherwise −1
~~~

The read operation is available through the module call, for example retrieved = sdm(address). Storage is available through sdm.write(address, word).

### Continuous distributed memory

ContinuousDM keeps the same distributed-address/content structure while replacing the hard threshold with a Gaussian radial-basis response. Its address rows are L2-normalized random vectors. For a unit-normalized query x, the implementation computes:

~~~text
dist²_m = 2 − 2(A_m · x)
y_m = exp(−dist²_m / (2σ²))
~~~

The content update is C ← C + y wᵀ. Retrieval computes s = Cᵀ y and L2-normalizes the result. The comparison script uses unit-normalized continuous addresses and words, sigma=0.2, and cosine similarity for the continuous output.

## Requirements

- Python 3.12 or newer
- PyTorch 2.13 or newer

The project metadata lives in pyproject.toml, and the repository includes uv.lock.

## Installation

Clone the repository and create the locked environment:

~~~bash
git clone https://github.com/SaM-0777/sparse_distributed_memory.git
cd sparse_distributed_memory
uv sync
~~~

The environment uses the Python version recorded in .python-version and installs the dependencies declared in pyproject.toml.

## Running the examples

Run the minimal entry point:

~~~bash
uv run python main.py
~~~

Run the single-pattern bipolar SDM experiment:

~~~bash
PYTHONPATH=. uv run python test/sdm_test.py
~~~

This script writes one random address/word pair, retrieves the word from the exact address, flips 15% of the address bits, and reports bit-match accuracy for both queries.

Run the incremental storage experiment:

~~~bash
PYTHONPATH=. uv run python test/sdm_bulk_test.py
~~~

This script stores 119 random patterns and prints network_stats() at 1, 25, 50, 100, and 119 stored words. The default configuration is N=256, U=256, M=1000, and H=110.

Run the bipolar-versus-continuous comparison:

~~~bash
PYTHONPATH=src:. uv run python test/continuous_vs_bipolar.py
~~~

The additional src entry in PYTHONPATH makes the comparison script’s direct continuous_dm import resolve to src/continuous_dm.py. The script prints a table over noise levels from 0.0 through 0.5, measuring bipolar bit-match percentage and continuous cosine similarity percentage.

The repository uses executable experiment scripts as its current test workflow. Each script draws fresh random data, so reported accuracies vary between runs.

## Python API example

Run this from the repository root:

~~~python
import torch

from src.sdm import SparseDistributedMemory


N = 256
U = 256

memory = SparseDistributedMemory(
    n_dim=N,
    u_dim=U,
    m_dim=1000,
    activation_radius_h=110,
    counter_range=15,
)

address = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1
word = torch.randint(0, 2, (U,), dtype=torch.float32) * 2 - 1

memory.write(address, word)
retrieved = memory(address)
accuracy = (retrieved == word).float().mean().item()

print(f"bit-match accuracy: {accuracy:.2%}")
memory.network_stats(target_fidelity=0.999)
~~~

The continuous variant follows the same write/read pattern with unit-normalized vectors:

~~~python
import torch
import torch.nn.functional as F

from src.continuous_dm import ContinuousDM


memory = ContinuousDM(n_dim=256, u_dim=256, m_dim=1000, sigma=0.2)
address = F.normalize(torch.randn(256), p=2, dim=0)
word = F.normalize(torch.randn(256), p=2, dim=0)

memory.write(address, word)
retrieved = memory(address)
~~~

## Capacity statistics

SparseDistributedMemory.network_stats() exposes the capacity heuristic used by the bulk experiment. For the supported target fidelities, the implementation uses ρ = 3.09 for 0.999, ρ = 2.33 for 0.99, and ρ = 1.64 for 0.95, then computes:

~~~text
τ_max = 1 / ρ²
T_max = floor(τ_max · M)
practical_max_words = floor(0.05 · M)
~~~

For M=1000 and the default target fidelity 0.999, the method reports a theoretical capacity of approximately 10.5% of M, 104 theoretical words, and 50 practical words. These values come directly from the formulas implemented in src/sdm.py.

## References

The bipolar model follows the address-space, thresholded hard-location, and distributed-content ideas developed in Kanerva’s SDM work. The Willshaw–Buneman–Longuet-Higgins paper provides an important earlier associative-memory reference.

1. Pentti Kanerva. *Sparse Distributed Memory*. MIT Press, 1988. ISBN 978-0-262-11132-4. [MIT Press bibliographic record via Google Books](https://books.google.com/books/about/Sparse_Distributed_Memory.html?id=I9tCr21-s-AC&output=html_text).
2. David J. Willshaw, O. P. Buneman, and H. C. Longuet-Higgins. “Non-Holographic Associative Memory.” *Nature*, 222, 960–962, 1969. [https://doi.org/10.1038/222960a0](https://doi.org/10.1038/222960a0).
3. David Rogers. *Kanerva’s Sparse Distributed Memory: An Associative Memory Algorithm Well-Suited to the Connection Machine*. NASA Contractor Report NASA-CR-185417, RIACS-TR-88.32, 1988. [NASA Technical Reports Server record and PDF](https://ntrs.nasa.gov/citations/19890017031).
4. James D. Keeler. “Comparison between Kanerva’s SDM and Hopfield-type neural networks.” *Cognitive Science*, 12(3), 299–329, 1988. [https://doi.org/10.1016/0364-0213(88)90026-2](https://doi.org/10.1016/0364-0213(88)90026-2).

## Citation

For this repository, use the following BibTeX entry and update the version field when publishing a tagged release:

~~~bibtex
@software{sam0777_sparse_distributed_memory,
  author  = {{SaM-0777}},
  title   = {sparse\_distributed\_memory},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/SaM-0777/sparse_distributed_memory}
}
~~~

For the underlying SDM method, cite Kanerva’s book:

~~~bibtex
@book{kanerva1988sparse,
  author    = {Kanerva, Pentti},
  title     = {Sparse Distributed Memory},
  publisher = {MIT Press},
  year      = {1988},
  isbn      = {978-0-262-11132-4},
  url       = {https://books.google.com/books/about/Sparse_Distributed_Memory.html?id=I9tCr21-s-AC&output=html_text}
}
~~~

## License

This project is licensed under the MIT License.

Copyright (c) 2026 SaM-0777

