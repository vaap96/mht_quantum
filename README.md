# mht_quantum

Research code for benchmarking quantum-inspired and classical solvers on:

- random 8-SAT instances (QAOA vs WalkSAT variants),
- graph-based DAP-style formulations (MWIS/MWC as WCNF/QUBO/Ising).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository File Map

### Root files

| File | Purpose |
|---|---|
| `README.md` | Project overview and file-by-file map. |
| `requirements.txt` | Python dependencies (NumPy/SciPy, plotting, `numba`, `networkx`, `python-sat`, Jupyter). |
| `test_dap.ipynb` | Notebook that runs the DAP flow end-to-end: generate graph instance, convert to WCNF, solve via exact RC2 and custom WalkSAT, then build QUBO. |
| `.DS_Store` | macOS Finder metadata file (not part of project logic). |

### `src/ksat/` (8-SAT pipeline)

| File | Purpose |
|---|---|
| `src/ksat/sat_utils.py` | Data generation utilities: random 8-SAT clauses, exact SAT solution search by brute force, Hamiltonian diagonal construction, train/test dataset builders. |
| `src/ksat/qaoa.py` | QAOA utilities for 8-SAT: linear-ramp parameterization, statevector simulation, full-QAOA training, LR-QAOA grid-search training. |
| `src/ksat/walksat.py` | High-performance WalkSAT implementations for 8-SAT, including a Numba kernel and a paper-style WalkSATlm kernel with make/break scoring. |
| `src/ksat/walksat_legacy.py` | Older/clearer (but slower) WalkSAT and WalkSATlm baselines kept for reference and comparison. |
| `src/ksat/evaluation.py` | Benchmark/evaluation routines for QAOA and WalkSAT variants, plus exponential scaling fit helper. |

### `src/mht/` (DAP / graph optimization helpers)

| File | Purpose |
|---|---|
| `src/mht/generate_dap.py` | Graph instance generation plus transformations: graph -> WCNF (Max-2-SAT), graph -> QUBO, and QUBO -> Ising/diagonal Hamiltonian. |
| `src/mht/classical_solvers_dap.py` | Classical solving backends for WCNF: exact RC2 wrapper and a weighted WalkSAT-style heuristic kernel. |
| `src/mht/qaoa_mht.py` | QAOA helper module for MHT/DAP workflows, including unified LR-QAOA training in supervised and unsupervised modes. |

### `ksat_implementation/` (experiment scripts/notebooks)

| File | Purpose |
|---|---|
| `ksat_implementation/main_benchmark_8sat.py` | Script version of the 8-SAT benchmark workflow (data generation, solver benchmarking, scaling estimation, plotting). |
| `ksat_implementation/main_benchmark_testbed_8sat.ipynb` | Notebook testbed for running the modular `src/ksat/*` benchmarking pipeline interactively. |
| `ksat_implementation/Final  LR QAOA vs QAOA vs walksat.ipynb` | Monolithic final experiment notebook containing full benchmark functions inline and producing comparative scaling plots. |

### `.idea/` (PyCharm project metadata)

| File | Purpose |
|---|---|
| `.idea/.gitignore` | PyCharm-specific ignore patterns for workspace-local files. |
| `.idea/misc.xml` | IDE project SDK/interpreter settings. |
| `.idea/modules.xml` | PyCharm module registration. |
| `.idea/mht_quantum.iml` | Module configuration (`.venv` exclusion, SDK binding). |
| `.idea/inspectionProfiles/Project_Default.xml` | Inspection/lint profile settings used by PyCharm. |
| `.idea/inspectionProfiles/profiles_settings.xml` | Inspection profile selection settings. |

## Typical Workflows

- 8-SAT benchmark (script): run `python ksat_implementation/main_benchmark_8sat.py`
- 8-SAT benchmark (notebook): open `ksat_implementation/main_benchmark_testbed_8sat.ipynb`
- DAP test notebook: open `test_dap.ipynb`
