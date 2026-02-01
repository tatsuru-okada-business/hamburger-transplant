#!/usr/bin/env python3
"""
Download organ and stitching weights from HuggingFace Hub.

Usage:
    python scripts/download_weights.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse


def main():
    parser = argparse.ArgumentParser(description="Download weights from HuggingFace Hub")
    parser.add_argument(
        "--repo",
        type=str,
        default="tatsuru-okada/hamburger-transplant-weights",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./weights/)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "weights"

    print("=" * 70)
    print("Downloading Hamburger Transplant Weights")
    print(f"From: {args.repo}")
    print(f"To:   {output_dir}")
    print("=" * 70)

    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(output_dir),
            allow_patterns=["organs/*", "stitching/*", "config.json", "README.md"],
        )
        print("\nDownload complete!")
        print(f"\nFiles in {output_dir}:")
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"\nError downloading from {args.repo}: {e}")
        print("\nTo set up weights manually:")
        print("  1. Create a 'weights/' directory in the project root")
        print("  2. Copy organ files to weights/organs/")
        print("  3. Copy stitching files to weights/stitching/")
        print("\nRequired files:")
        print("  weights/organs/r1_math.pt        (Math organ from DeepSeek-R1)")
        print("  weights/organs/code_organ.pt     (Code organ from Qwen2.5-Coder)")
        print("  weights/organs/qwen3_japanese.pt  (Japanese organ from Qwen3)")
        print("  weights/stitching/math_stitching_1epoch.pt")
        print("  weights/stitching/code_stitching_1epoch.pt")
        print("  weights/stitching/japanese_stitching_1epoch.pt")
        sys.exit(1)


if __name__ == "__main__":
    main()
