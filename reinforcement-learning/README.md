## About

1\. Perform exploratory data analysis (EDA): visualize trends, volatility, and return distributions.

2\. Apply unsupervised learning: dimensionality reduction (PCA, autoencoders) to build compact market state representations.

Afterwards we are going to use the unsupervised results as state features in an RL environment.

3\. Implement a simple RL trading agent: Environment: NVDA daily returns with transaction costs. Action space: {-1 = short, 0 = hold, +1 = long}.

Reward:

$$r_t^{\text{net}} = a_{t-1} \cdot \text{return}_t - \text{cost} \cdot |a_t - a_{t-1}|$$

Policy: policy-gradient method, PPO

4\. Evaluate performance: Primary: cumulative PnL vs. buy-and-hold baseline. Secondary: average return, max drawdown, hit ratio, turnover.


Dataset: Daily granularity data for Nvidias (NVDA) stock from 2010 to this date. You can find the dataset [here](https://pypi.org/project/yfinance/).

## Repository Structure

```text
├── ppo/
│   ├── network.py                  # Actor/critic network definitions
│   ├── nvda_env.py                 # NVDA trading gym environment
│   └── ppo.py                      # PPO training loop
├── .gitignore                      # Git ignore rules
├── .python-version                 # Python version for uv
├── README.md                       # Project documentation
├── autoencoder.py                  # Autoencoder training and encoding helpers
├── evaluation.py                   # Policy evaluation utility
├── pyproject.toml                  # Project metadata and dependency constraints
├── unsupervised_rl_learning.ipynb  # Main notebook
└── uv.lock                         # Locked dependency versions
```

## Getting Started

We use [uv](https://docs.astral.sh/uv/) for dependency management. Dependency constraints live in `pyproject.toml` and exact resolved versions are pinned in `uv.lock`, so every install reproduces the same environment.

### Setup

1) Install `uv` (see the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
2) Clone the repository and create the project environment from the lockfile:

```bash
git clone <REPO_URL>
cd <REPO_DIR>
uv sync
```

`uv sync` reads `.python-version` to pick the right Python (3.12), creates a `.venv/` in the project root, and installs every package at the version recorded in `uv.lock`.

### Running the notebook

```bash
uv run jupyter notebook unsupervised_rl_learning.ipynb
```

Then run all cells from top to bottom.

## Author

Moritz Ehlert