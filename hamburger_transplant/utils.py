"""
Utility functions for Hamburger Transplantation.
"""

import torch
from pathlib import Path


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_weights_dir() -> Path:
    """Return the weights directory (organs + stitching)."""
    root = get_project_root()
    weights = root / "weights"
    if weights.exists():
        return weights
    # Fallback: check for organs/ directly in project root
    if (root / "organs").exists():
        return root
    return weights


def get_organs_dir() -> Path:
    """Return organs directory."""
    return get_weights_dir() / "organs"


def get_stitching_dir() -> Path:
    """Return stitching weights directory."""
    return get_weights_dir() / "stitching"


def ensure_organ_exists(organ_path: Path, organ_name: str = "organ") -> None:
    """Check that an organ file exists, raise helpful error if not."""
    if not organ_path.exists():
        raise FileNotFoundError(
            f"Organ file not found: {organ_path}\n"
            f"Please download weights first:\n"
            f"  python scripts/download_weights.py\n"
            f"Or manually place {organ_name} at: {organ_path}"
        )


def ensure_stitching_exists(stitching_path: Path, organ_name: str = "organ") -> None:
    """Check that a stitching file exists, raise helpful error if not."""
    if not stitching_path.exists():
        raise FileNotFoundError(
            f"Stitching weights not found: {stitching_path}\n"
            f"Train stitching first:\n"
            f"  python scripts/train_{organ_name}_stitching.py"
        )
