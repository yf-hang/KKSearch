# KK Eigenfunction Relation Search

This project uses neural networks to explore identities among Kaluza–Klein (KK) 
wavefunctions. Compactification of a higher-dimensional theory produces a tower of 
four-dimensional modes, each with a mass and an extra-dimensional profile. 
In a warped geometry, the bulk equations and boundary conditions determine these profiles. 
Weighted overlaps of the profiles and their derivatives enter the interaction couplings of 
the four-dimensional effective theory.

The physical setup follows [arXiv:2504.05199](https://arxiv.org/pdf/2504.05199v3) and [arXiv:2406.12713](https://arxiv.org/pdf/2406.12713v3), 
and includes scalar matter and gravitational fields. 
The table distinguishes the matter profiles from the graviton and its vector and 
scalar Goldstone partners. Their mode equations, normalizations and boundary conditions 
constrain the possible overlap relations and can be found in the references provided above.

| Eigenfunction | Corresponding Field |
|---|---|
| `s` | Scalar matter modes |
| `u` | KK gravitons, including the massless graviton zero mode |
| `v` | Gravitational vector Goldstone partners; no zero mode |
| `w` | Gravitational scalar Goldstone partners; the zero mode is the radion |

A gated recurrent unit (GRU) network searches a mechanically generated dictionary of overlap integrals for integer
coefficient relations. It learns from numerical training data without receiving target equations. Candidates are
selected on validation data and frozen before checks on held-out backgrounds, unseen KK modes and independently
recomputed profiles. The current implementation considers products of six profiles and aims to identify explicit
relations not displayed in the references. Numerical validity and literature novelty are assessed separately.

## Run

From the project root:

```bash
.venv/bin/python main.py
```

Without a subcommand, this starts six-point training; `python main.py run` is equivalent. In PyCharm use script `main.py`, leave arguments empty, use the project root and interpreter `.venv/bin/python`. Installation also provides `neural-search`.

For a fresh environment (Python 3.10 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Repository Structure

```text
|-- main.py         # Entry point: run, sample, verify and report; coordinate the pipeline
|-- config.json     # Experiment settings: seeds, search budget, data and solver accuracy
|-- physics.py      # Solve field profiles, build the integral dictionary and generate data
|-- model.py        # Conditional GRU, coefficient distributions, sampling and model loading
|-- train.py        # Matrix-only neural training, candidate scoring and self-replay
|-- validation.py   # Numerical checks and post-freeze analysis of published premises
|-- plot_tool.py    # PNG total-loss plot and readable experiment reports
|-- utils.py        # Configuration loading, JSON writing and SHA-256 file checks
|-- test_core.py    # 16 checks of physics, network calculations and training isolation
|-- pyproject.toml  # Project metadata, dependencies, packaging and CLI installation
```

To follow a run, start with `main.py`: it reads `config.json`, calls `physics.py` to generate data, runs `train.py` with the network from `model.py`, then invokes `validation.py` and `plot_tool.py`.

`test_core.py` checks the physical solvers, neural probability calculations, checkpoint loading and training-input isolation. These tests run separately from the search.

## Dictionary, network and scope

Six external profiles drawn mechanically from `s,u,v,w` give **84 profile multisets**. Each has **396 primitives**, totaling **33,264 terms**. Every primitive contains six profiles, warp weight 1, 3, 5 or 7, and zero, one or two derivatives with at most one derivative per factor. Background/mass factors from `q,b,M0,…,M5` complete mass dimension two. No target equation is preassembled.

A candidate combines terms within one fixed set of external-profile types, called a context. Focused search covers **21 eligible mixed contexts** with two or four scalar profiles, representing one or two conjugate pairs and the remaining gravitational states. Cross-context identities and internal KK spectral sums are outside this grammar.

The network in `model.py` is a conditional GRU with **4,940,384 parameters** and a 512-dimensional hidden state. A learned initial state and an `84×512` context embedding condition the network on the numeric context ID. At each decoding stage, a 128-dimensional token embedding and a 128-dimensional stage embedding are concatenated into a 256-dimensional input.

The same GRUCell is used for three successive stages. First, it selects one of the 396 coefficient positions as a nonzero anchor. Second, it selects the anchor's value from the positive integers 1 through 8. Third, it produces all 396 coefficient distributions in parallel, with 17 possible values from −8 to 8 for each term. The anchor coefficient is fixed to the selected positive value, while zero omits any other term. The remaining coefficients are therefore generated after three recurrent steps, rather than one recurrent step per coefficient.

The search evaluates categorical samples, conditional modes and rounded coefficient means as actual integer vectors. Numerical training matrices enter the objective used to update the network; the decoding inputs contain only the context ID and control tokens. Validation selects candidates, and held-out tests follow candidate freezing.

The isolated learner receives only numeric training data, generic settings and two computational source files (`model.py` and `train.py`). It cannot read the papers, solver, novelty reviewer, old model or test data. SVD, nullspace solving and linear fitting do not generate coefficient answers. Training starts from fresh weights; training/validation candidates are frozen before held-out and independent Bessel/shooting checks.

Initial proposals cover all 84 contexts. Focused mixed contexts are then selected randomly for up to 1,500 consecutive updates, advancing after numerical hits. The total budget is 9,000 updates; this does not establish extensive exploration of every context. The only configuration is `config.json`; there is no small model configuration.

## Training objective

Total loss is minimized by Adam. It combines a differentiable residual guide, built from coefficient means and variances, with the seven terms below. 
Their weights are set in `config.json`. Here, μ denotes a coefficient mean and K is the expected number of nonzero coefficients.

| Term | Current weight | Purpose |
|---|---|---|
| Integer-mean penalty | Ramps to 0.05 over 2,000 updates | Penalizes `sin²(πμ)` to move coefficient means toward integers and aid rounded-mean proposals. Samples are already integer-valued. |
| Mean-magnitude penalty | 0.01 | Penalizes the average absolute value of the coefficient means, favoring smaller magnitudes. |
| Sparsity penalty | 0.001 | Penalizes the expected number of nonzero terms. |
| Minimum-support penalty | 1 | Adds `max(2-K, 0)²` when expected support K is below two. Actual candidates also face a separate two-term minimum. |
| Control entropy | 0.1 | Subtracts anchor and magnitude entropy to discourage premature concentration on a few choices. |
| Control policy gradient | 0.1 | Uses integer-proposal rewards and the residual guide to improve anchor and magnitude choices. |
| Self-replay | 0.1 | Raises the probability of good proposals previously generated by the network, with normalization by dictionary size. Replay starts empty. |

These terms guide search without supplying target equations. Mean-magnitude and sparsity penalties have partly overlapping purposes; their individual necessity has not been established by ablation experiments. Some terms enter with a negative sign, so total loss can be negative. Numerical acceptance is determined by the actual integer residuals and independent verification.

## Checks

```bash
.venv/bin/python main.py verify --high-precision
.venv/bin/python main.py sample --count 20
.venv/bin/python main.py report
.venv/bin/python -m unittest -v test_core
```

