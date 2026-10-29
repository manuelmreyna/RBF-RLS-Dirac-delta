# Physics-Informed Neural Networks and Radial Basis Functions for PDEs with Dirac Delta Sources

Code, data, and numerical results accompanying:

> Manuel M. Reyna and Alexandre M. Tartakovsky. Physics-Informed Neural Networks and Radial Basis Functions for PDEs with Dirac Delta Sources.

This repository compares **radial basis function residual least squares
(RBF-RLS)** with **physics-informed neural networks (PINNs)** for
advection–diffusion-type PDEs with Dirac-delta sources, in both forward and
inverse (parameter-estimation) settings. Four test problems are covered.

| # | Problem | Unknowns in the inverse problem | RBF-RLS | PINN |
|---|---------|---------------------------------|:-------:|:----:|
| 1 | 1D advection–diffusion equation (ADE) with an instantaneous point release (Dirac initial condition) | `v`, `D` | ✔ | – |
| 2 | 1D ADE applied with sudden upstream boundary condition to replicate measurements by Levy and Berkowitz (2003)  (`homo1.txt`) | `v`, `D` | ✔ | ✔ |
| 3 | ADE with first-order exchange with an immobile phase, applied to a field breakthrough curve (Antietam Creek) | `M0`, `v`, `D`, `βk`, `k` | ✔ | – |
| 4 | 1D diffusion with a local (Dirac) forcing term, noisy observations | `k` | ✔ | ✔ |

## Repository contents

| File | Description |
|------|-------------|
| `RBF_RLS_problems_1_3_4.ipynb` | RBF-RLS forward and inverse solvers for Problems 1, 3 and 4, including the hyperparameter studies, optimizer comparison (derivative-free vs. gradient vs. gradient + Hessian) and figures |
| `RBF_RLS_problem_2.ipynb` | RBF-RLS forward and inverse solver for Problem 2, compared against the PINN results |
| `PINN_problem_2.ipynb` | PINN forward and inverse solver for Problem 2 |
| `PINN_problem_4.ipynb` | PINN for Problem 4, with Bayesian optimisation of the architecture and loss weights |
| `invlap.py` | Numerical inverse Laplace transform (de Hoog algorithm), used to compute reference solutions. Third-party code, see [Acknowledgements](#third-party-code-and-data) |
| `homo1.txt` | Measured breakthrough curve measured by Levy and Berkowitz (2003) used in Problem 2 (columns: `time`, `Measured`, `CTRW`, `ADE`) |

## Generated contents

| File | Description |
|------|-------------|
| `PINN_problem_2_est_params.npy` | Parameters `[v, D]` estimated by the PINN inverse run (Problem 2) |
| `PINN_problem_2_fwd_output.npy`, `PINN_problem_2_inv_output.npy` | PINN solution fields on the 201 × 201 (t, x) evaluation grid (forward and inverse runs) |
| `PINN_problem_2_model.pt`, `PINN_problem_2_model_inverse.pt` | Trained forward and inverse PINN checkpoint (model, optimizer, scheduler state and loss history) |
| `RBF_local_forcing.npy` | RBF-RLS solution field for Problem 4 (200 × 200), consumed by `PINN_problem_4.ipynb` |
| `model_snapshots_PINN_local_forcing.pt` | PINN weight snapshots saved during the Problem 4 training run |

Fields are stored with time on axis 0 and space on axis 1, in non-dimensional
coordinates on the unit square.

## Installation

Python **3.11 or newer** is required.

```bash
git clone https://github.com/[user]/[repo].git
cd [repo]

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Optional but recommended for the PINN notebooks: a CUDA build of PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
jupyter lab
```

The notebooks import `invlap` directly, so launch Jupyter from the repository
root (or keep `invlap.py` next to the notebooks).

## Reproducing the results

The notebooks are meant to be run top to bottom. Two of them depend on files
produced by another notebook, so mind the order:

```
PINN_problem_2.ipynb ──► PINN_problem_2_{fwd_output,inv_output,est_params}.npy
                                      │
                                      ▼
                        RBF_RLS_problem_2.ipynb      (needs homo1.txt, invlap.py)

RBF_RLS_problems_1_3_4.ipynb ──► RBF_local_forcing.npy
                                      │
                                      ▼
                        PINN_problem_4.ipynb
```

To skip the long PINN training, use the precomputed `.npy` / `.pt` files in
this repository: `RBF_RLS_problem_2.ipynb` and `PINN_problem_4.ipynb` run
directly from them.

### Approximate runtimes

Measured on a single NVIDIA RTX A2000 (PINNs) and a Ryzen 7 4700U CPU (RBF-RLS).
Your timings will differ.

| Notebook | Dominant cost | Time |
|----------|---------------|------|
| `PINN_problem_2.ipynb` | Forward PINN, 1.25 M Adam epochs | ≈ 6.5 h |
| `PINN_problem_4.ipynb` | Bayesian optimisation of the PINN setup | ≈ 3.7 h |
| `RBF_RLS_problem_2.ipynb` | Matrix assembly and inverse optimisation | ≈ 7 min |
| `RBF_RLS_problems_1_3_4.ipynb` | Hyperparameter sweeps (Problem 1), optimizer comparisons (Problem 3), Bayesian optimisation (Problem 4) | several hours in total; the individual sweeps range from ~10 s to ~45 min |

### Reproducibility notes

- Random seeds are fixed (`seed = 42`) in the PINN notebooks. GPU kernels and
  the unseeded Latin-hypercube sampler used in the Problem 4 Bayesian
  optimisation mean that PINN results are reproducible only up to
  run-to-run variation, not bit-for-bit.
- The RBF-RLS solvers are deterministic apart from the randomly perturbed
  initial guesses and noise realisations, which use fixed NumPy seeds.

## Third-party code and data

- **`invlap.py`** is a Python translation of the MATLAB function `INVLAP.M`
  by K. J. Hollenbeck (1998), implementing the de Hoog et al. (1982)
  algorithm. If you use it, please cite:
  - Hollenbeck, K. J. (1998). *INVLAP.M: A MATLAB function for numerical
    inversion of Laplace transforms by the de Hoog algorithm.*
  - de Hoog, F. R., Knight, J. H., and Stokes, A. N. (1982). An improved
    method for numerical inversion of Laplace transforms. *SIAM J. Sci. Stat.
    Comput.*, 3, 357–366.
- **`homo1.txt`**: Levy, M., & Berkowitz, B. (2003). Measurement and analysis of non-Fickian dispersion in heterogeneous porous media. Journal of Contaminant Hydrology, 64(3–4), 203–226. https://doi.org/10.1016/s0169-7722(02)00204-8
- **Antietam Creek breakthrough curve (Problem 3)**, embedded in
  `RBF_RLS_problems_1_3_4.ipynb`: Nordin, C.F., and Sabol, G.V., 1974, Empirical data on longitudinal dispersion in rivers: U.S. Geological Survey Water-Resources Investigations Report 74-20, vii, 332 p., https://doi.org/10.3133/wri7420. 
  Can be downloaded from the TIERRAS dataset: Rodríguez, L., Aghababaei, M,. Tunby, P., Abusang, A., Tartakovsky, A., Carroll, K., Ginn, T., & González-Pinzón, R. (2025). TIERRAS Tracer Injection Experiments in RiveRs And Streams (2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15793577.

These items remain subject to their original terms and are not covered by the
license below unless their authors permit it.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{reyna2026physicsinformedneuralnetworksradial,
      title={Physics-Informed Neural Networks and Radial Basis Functions for PDEs with Dirac Delta Sources}, 
      author={Manuel Reyna and Alexandre Tartakovsky},
      year={2026},
      eprint={2606.12735},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.12735}, 
}
```

## License

The code in this repository is released under the [MIT License](LICENSE),
except for the third-party material listed above.
