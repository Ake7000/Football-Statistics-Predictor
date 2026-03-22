# shared_utils.py
# Utility functions shared across all optimizer and classifier scripts.
# Contains: seeding, GPU detection, path helpers, metrics,
#           and shared MLP building blocks (get_activation, snap_to_choices,
#           build_layer_sizes, make_torch_optimizer, ResidualMLP).

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# ==  REPRODUCIBILITY  ========================================================
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=False keeps cuDNN fast; benchmark=True picks fastest kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# ==  DEVICE DETECTION  =======================================================
# =============================================================================

def get_torch_device() -> torch.device:
    """Return CUDA device if available, else CPU.  Prints a one-line status."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using CUDA GPU: {name}")
        return torch.device("cuda")
    print("[WARN] No GPU found. Running on CPU.")
    return torch.device("cpu")


def get_xgb_tree_method() -> str:
    """Always returns 'hist' — the only valid tree_method in XGBoost ≥ 2.0.
    GPU selection is done via get_xgb_device(), not tree_method."""
    return "hist"


def get_xgb_device() -> str:
    """
    Return 'cuda' if a CUDA GPU is visible, else 'cpu'.
    XGBoost ≥ 2.0 uses device= instead of tree_method= for GPU/CPU selection.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] XGBoost will run on GPU: {name}")
        return "cuda"
    print("[INFO] No GPU detected. XGBoost will run on CPU.")
    return "cpu"


# =============================================================================
# ==  PATHS & DIRECTORIES  ====================================================
# =============================================================================

def timestamp_slug() -> str:
    """Return a compact timestamp string, e.g. '20260226-153042-047123'."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def make_run_dir(artifacts_root: Path, csv_path: Path, variant: str, suffix: str = "") -> Path:
    """
    Create and return a timestamped run directory under:
        <artifacts_root>/<variant>/<csv_stem>__<timestamp>[<suffix>]/

    Args:
        artifacts_root: e.g. ARTIFACTS_MLP_ROOT or ARTIFACTS_XGB_ROOT
        csv_path:       path to the training CSV (used for the dir name)
        variant:        feature-table variant string (e.g. 'raw', 'mean')
        suffix:         optional suffix appended after the timestamp (e.g. '__cw_sqrt')
    """
    safe_stem = csv_path.stem  # filename without extension
    run_dir = artifacts_root / variant / f"{safe_stem}__{timestamp_slug()}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# ==  METRICS  ================================================================
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


# =============================================================================
# ==  SHARED MLP BUILDING BLOCKS  =============================================
# =============================================================================
# These functions are identical across all optimizer_*_torch.py and
# classifier_*_torch.py files.  Defined once here; imported everywhere.

def get_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name (relu/gelu/selu/elu/swish)."""
    return {
        "relu":  nn.ReLU,
        "gelu":  nn.GELU,
        "selu":  nn.SELU,
        "elu":   nn.ELU,
        "swish": nn.SiLU,   # SiLU ≡ Swish
    }.get(name.lower(), nn.ReLU)()


def snap_to_choices(v: int, choices: List[int]) -> int:
    """Return the element of `choices` closest to `v`."""
    return min(choices, key=lambda c: abs(c - v))


def build_layer_sizes(params: Dict, units_choices: List[int]) -> List[int]:
    """
    Reconstruct the hidden-layer sizes list from an Optuna params dict.

    Supports both trial.params (during search) and best_params dicts (for
    retrain).  The parameterisation is:
        n_hidden     — number of hidden layers
        base_units   — size of the first hidden layer
        mult_1 …     — multiplicative factor applied to grow/shrink subsequent
                       layers  (values snapped to the nearest choice in
                       units_choices).
    """
    n_hidden   = int(params.get("n_hidden",   1))
    base_units = int(params.get("base_units", units_choices[0]))
    sizes = [base_units]
    for k in range(1, n_hidden):
        mult = float(params.get(f"mult_{k}", 1.0))
        sizes.append(snap_to_choices(int(round(sizes[-1] * mult)), units_choices))
    return sizes


def make_torch_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """Create an Adam / AdamW / NAdam optimizer by name."""
    name = name.lower()
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


class ResidualMLP(nn.Module):
    """
    Pre-norm residual MLP backbone, optionally with a linear output head.

    Architecture:
        [LayerNorm → Linear → Activation → Dropout] × n_layers
        with residual skip connections added every 2 linear layers.

    Parameters
    ----------
    input_dim   : int  — number of input features.
    layer_sizes : list[int]  — hidden-layer widths.
    activation  : str  — activation name (relu / gelu / selu / elu / swish).
    dropout     : float — dropout probability (0 = disabled).
    out_dim     : int | None
        * None  — no head; forward() returns the raw backbone vector
                  (shape (B, layer_sizes[-1])).  Use for multi-head models.
        * int   — applies a final nn.Linear(last_hidden, out_dim).
    squeeze     : bool
        * True  — squeeze the last dim (useful for out_dim=1 regression).
        * False — return (B, out_dim) logits (useful for classification).

    Attribute
    ---------
    out_dim : int   — backbone output dimension (= layer_sizes[-1]).
                      Available even when no head is attached.

    Compatible with MLPEncoder in shared_sequence.py (same attribute names
    and identical forward logic → existing .pt checkpoints load cleanly).
    """

    def __init__(
        self,
        input_dim:   int,
        layer_sizes: List[int],
        activation:  str,
        dropout:     float,
        out_dim:     Optional[int] = None,
        squeeze:     bool          = False,
    ) -> None:
        super().__init__()
        self.norms:   nn.ModuleList = nn.ModuleList()
        self.linears: nn.ModuleList = nn.ModuleList()
        self.acts:    nn.ModuleList = nn.ModuleList()
        self.drops:   nn.ModuleList = nn.ModuleList()
        self.projs:   nn.ModuleList = nn.ModuleList()   # one per complete 2-layer block
        prev         = input_dim
        block_in_dim = input_dim
        for i, units in enumerate(layer_sizes):
            self.norms.append(nn.LayerNorm(prev))
            self.linears.append(nn.Linear(prev, units))
            self.acts.append(get_activation(activation))
            self.drops.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
            if i % 2 == 0:
                block_in_dim = prev
            if i % 2 == 1:
                self.projs.append(
                    nn.Linear(block_in_dim, units, bias=False)
                    if block_in_dim != units else nn.Identity()
                )
            prev = units
        self.out_dim   = prev                                      # backbone output dim
        self.head      = nn.Linear(prev, out_dim) if out_dim is not None else None
        self._n_layers = len(layer_sizes)
        self._squeeze  = squeeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip     = x
        proj_idx = 0
        for i in range(self._n_layers):
            if i % 2 == 0:
                skip = x
            x = self.drops[i](self.acts[i](self.linears[i](self.norms[i](x))))
            if i % 2 == 1:
                x = x + self.projs[proj_idx](skip)
                proj_idx += 1
        if self.head is not None:
            x = self.head(x)
            if self._squeeze:
                x = x.squeeze(-1)
        return x
