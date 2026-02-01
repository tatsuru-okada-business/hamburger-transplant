# Hamburger Transplantation

Lightweight LLM capability enhancement via neural organ insertion with stitching layers.

## Overview

Hamburger Transplantation enhances large language models by additively inserting frozen MLP layers ("organs") extracted from specialist models, connected via thin learnable stitching layers. Unlike fine-tuning or LoRA, this method requires only **30 seconds of training** with **200 samples** on a consumer laptop.

**Key results:**
- HumanEval: 80.0% → **84.0%** (+4pp) with dual organ (Math + Code)
- JCommonsenseQA: 82.0% → **92.0%** (+10pp) with triple organ
- Cross-scale transplantation: organs from models up to 2x the host's hidden dimension
- All experiments on a single MacBook Pro (M4 Pro, 24GB)

## Architecture

```
Host Layer 0-4:  Base Transformer
                 ── MATH ORGAN (DeepSeek-R1) ──  ← Layer 5
Host Layer 5-10: Base Transformer
                 ── CODE ORGAN (Qwen2.5-Coder) ── ← Layer 11
Host Layer 11-16: Base Transformer
                 ── JAPANESE ORGAN (various) ──   ← Layer 17
Host Layer 17-27: Base Transformer
```

Each organ is connected via residual path:
```
h' = h + α · stitch_out(MLP_organ(LayerNorm(stitch_in(h))))
```

## Quick Start

```bash
# Install
pip install -e .

# Download weights
python scripts/download_weights.py

# Train stitching layers (~30s each)
python scripts/train_math_stitching.py
python scripts/train_code_stitching.py

# Run benchmark
python scripts/benchmark_fair.py --num-problems 50
```

## Project Structure

```
hamburger-transplant-public/
├── hamburger_transplant/      # Core module
│   ├── model.py               # HamburgerModelV2, StitchingLayer, OrganLayerV2
│   └── utils.py               # Device detection, path helpers
├── scripts/
│   ├── download_weights.py    # Download organ & stitching weights
│   ├── train_math_stitching.py
│   ├── train_code_stitching.py
│   └── benchmark_fair.py      # Fair 4-condition HumanEval benchmark
├── configs/                   # Organ configuration files (JSON)
├── paper/                     # Paper (LaTeX + PDF + Japanese translation)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.37
- 24GB+ RAM (Apple Silicon or CUDA GPU)

## Paper

**Hamburger Transplantation: Lightweight Additive Capability Enhancement via Neural Organ Insertion with Stitching Layers**

- [English (PDF)](paper/hamburger_transplant.pdf)
- [English (LaTeX source)](paper/hamburger_transplant.tex)
- [Japanese (Markdown)](paper/hamburger_transplant_ja.md)

## Citation

```bibtex
@article{okada2026hamburger,
  title={Hamburger Transplantation: Lightweight Additive Capability Enhancement via Neural Organ Insertion with Stitching Layers},
  author={Okada, Tatsuru},
  year={2026}
}
```

## Author

**Tatsuru Okada** — Independent Researcher

- GitHub: [@tatsuru-okada-business](https://github.com/tatsuru-okada-business)
- Company: [Plusing Inc.](https://plusing.inc/)

## Support

If you find this research useful, consider supporting continued development:

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/tatsuru-okada-business)

## License

MIT License. See [LICENSE](LICENSE) for details.
