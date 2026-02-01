#!/usr/bin/env python3
"""
Train Stitching Layers for Math Organ (1 epoch only).

Usage:
    python scripts/train_math_stitching.py
"""

import sys
from pathlib import Path

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
    """Train stitching layers for exactly 1 epoch on WikiText."""
    print("\n" + "=" * 70)
    print("Training Math Organ Stitching (1 EPOCH)")
    print("=" * 70)

    trainable_params = model.get_trainable_params()
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Use WikiText for general language
    print(f"\nLoading WikiText data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"][:num_samples * 5] if len(t.strip()) > 50][:num_samples]
    print(f"Loaded {len(texts)} samples")

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(texts, desc="Training (1 epoch)")

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
    print("MATH ORGAN STITCHING TRAINING (1 EPOCH)")
    print(f"Device: {device}")
    print("=" * 70)

    math_organ = [
        InsertedOrgan(
            name="math",
            insert_after_layer=5,
            organ_path=str(organs_dir / "r1_math.pt"),
            scale=0.1,
        ),
    ]

    model = HamburgerModelV2(organ_insertions=math_organ, device=device)

    train_stitching_1epoch(model, num_samples=200, lr=1e-3, max_length=128)

    model.save_stitching(str(stitching_dir / "math_stitching_1epoch.pt"))

    # Quick test
    print("\n" + "=" * 70)
    print("Quick test after training...")
    print("=" * 70)

    model.eval()
    prompts = [
        "What is 15 + 27?",
        "If a = 5 and b = 3, what is a * b + a?",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate(prompt, max_new_tokens=100)
        print(f"Response: {response[:200]}")


if __name__ == "__main__":
    main()
