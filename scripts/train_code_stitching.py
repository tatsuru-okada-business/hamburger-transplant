#!/usr/bin/env python3
"""
Train Stitching Layers for Code Organ (1 epoch only).

Usage:
    python scripts/train_code_stitching.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from hamburger_transplant.model import HamburgerModelV2, InsertedOrgan
from hamburger_transplant.utils import get_device, get_organs_dir, get_stitching_dir


def train_stitching_1epoch(
    model: HamburgerModelV2,
    num_samples: int = 200,
    lr: float = 1e-3,
    max_length: int = 128,
):
    """Train stitching layers for exactly 1 epoch on code data."""
    print("\n" + "=" * 70)
    print("Training Stitching Layers (1 EPOCH ONLY)")
    print("=" * 70)

    trainable_params = model.get_trainable_params()
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Load Code data
    print(f"\nLoading code data (codeparrot-clean)...")
    code_dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

    code_texts = []
    for item in code_dataset:
        content = item.get("content", "")
        if 100 < len(content) < 2000:
            if "def " in content or "class " in content:
                code_texts.append(content)
                if len(code_texts) >= num_samples:
                    break

    print(f"Loaded {len(code_texts)} code samples")

    # 1 epoch only
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(code_texts, desc="Training (1 epoch)")

    for text in pbar:
        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(model.device)

        if inputs["input_ids"].shape[1] < 10:
            continue

        optimizer.zero_grad()

        outputs = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    print(f"\n1 Epoch average loss: {avg_loss:.4f}")
    return avg_loss


def main():
    device = get_device()
    organs_dir = get_organs_dir()
    stitching_dir = get_stitching_dir()
    stitching_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CODE ORGAN STITCHING TRAINING (1 EPOCH)")
    print(f"Device: {device}")
    print("=" * 70)

    code_organ = [
        InsertedOrgan(
            name="code",
            insert_after_layer=11,
            organ_path=str(organs_dir / "code_organ.pt"),
            scale=0.1,
        ),
    ]

    model = HamburgerModelV2(organ_insertions=code_organ, device=device)

    train_stitching_1epoch(model, num_samples=200, lr=1e-3, max_length=128)

    model.save_stitching(str(stitching_dir / "code_stitching_1epoch.pt"))

    # Quick test
    print("\n" + "=" * 70)
    print("Quick test after 1 epoch training...")
    print("=" * 70)

    model.eval()
    prompts = [
        "What is 2 + 2?",
        "Write a function to check if a number is prime.",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate(prompt, max_new_tokens=100)
        print(f"Response: {response[:200]}")


if __name__ == "__main__":
    main()
