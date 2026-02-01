#!/usr/bin/env python3
"""
Fair 4-condition benchmark: separates Organ effect from Prompt effect.

Conditions:
  1. Base  + Simple prompt
  2. Base  + Reasoning prompt
  3. DUAL  + Simple prompt
  4. DUAL  + Reasoning prompt

All conditions use max_new_tokens=512 for fairness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import re
import gc
import random
import time
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from hamburger_transplant.model import HamburgerModelV2, InsertedOrgan
from hamburger_transplant.utils import get_device, get_organs_dir, get_stitching_dir


MAX_NEW_TOKENS = 512  # Unified for all conditions


def extract_code(text: str) -> str:
    """Extract Python code from model output."""
    code_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    func_pattern = r'(def\s+\w+.*?)(?=\ndef\s|\nclass\s|\Z)'
    matches = re.findall(func_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def make_prompt(problem_prompt: str, use_reasoning: bool) -> list:
    """Create chat messages for a HumanEval problem."""
    if use_reasoning:
        return [{
            "role": "user",
            "content": (
                f"Read the function specification carefully. "
                f"Pay close attention to the examples.\n\n"
                f"{problem_prompt}\n\n"
                f"Think step by step about what the function should do, then implement it."
            ),
        }]
    else:
        return [{
            "role": "user",
            "content": f"Complete this Python function. Only output the code, no explanations.\n\n{problem_prompt}",
        }]


def run_humaneval(model_obj, tokenizer, problems, device, use_reasoning):
    """Run HumanEval problems and return results."""
    correct = 0
    results = []

    for problem in tqdm(problems, desc=f"{'Reasoning' if use_reasoning else 'Simple'}"):
        task_id = problem['task_id']
        prompt = problem['prompt']
        test = problem['test']
        entry_point = problem['entry_point']

        messages = make_prompt(prompt, use_reasoning)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        generated_code = extract_code(response)

        try:
            full_code = prompt + generated_code
            exec_globals = {}
            exec(full_code, exec_globals)
            exec(test, exec_globals)
            check_func = exec_globals.get('check')
            if check_func:
                check_func(exec_globals[entry_point])
            correct += 1
            passed = True
        except Exception:
            passed = False

        status = "pass" if passed else "FAIL"
        print(f"  {task_id}: {status}")
        results.append({"task_id": task_id, "passed": passed})

    return correct, results


def run_base_conditions(problems, device):
    """Run base model (no organs) with both prompt types."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading BASE model (Qwen2.5-7B-Instruct, no organs)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Condition 1: Base + Simple
    print("\n" + "-" * 70)
    print("Condition 1: Base + Simple prompt")
    print("-" * 70)
    t0 = time.time()
    base_simple, base_simple_results = run_humaneval(model, tokenizer, problems, device, use_reasoning=False)
    t_base_simple = time.time() - t0

    # Condition 2: Base + Reasoning
    print("\n" + "-" * 70)
    print("Condition 2: Base + Reasoning prompt")
    print("-" * 70)
    t0 = time.time()
    base_reason, base_reason_results = run_humaneval(model, tokenizer, problems, device, use_reasoning=True)
    t_base_reason = time.time() - t0

    # Cleanup base model to free memory
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "base_simple": {"correct": base_simple, "time": t_base_simple, "results": base_simple_results},
        "base_reason": {"correct": base_reason, "time": t_base_reason, "results": base_reason_results},
    }


def run_dual_conditions(problems, device):
    """Run DUAL organ model with both prompt types."""
    organs_dir = get_organs_dir()
    stitching_dir = get_stitching_dir()

    dual_organs = [
        InsertedOrgan(name="math", insert_after_layer=5, organ_path=str(organs_dir / "r1_math.pt"), scale=0.1),
        InsertedOrgan(name="code", insert_after_layer=11, organ_path=str(organs_dir / "code_organ.pt"), scale=0.1),
    ]

    print("\nLoading DUAL organ model (Math@5 + Code@11)...")
    model = HamburgerModelV2(organ_insertions=dual_organs, device=device)

    # Load stitching weights
    for organ_name in ["math", "code"]:
        stitching_path = stitching_dir / f"{organ_name}_stitching_1epoch.pt"
        if stitching_path.exists():
            state = torch.load(stitching_path, map_location='cpu', weights_only=False)
            if organ_name in state:
                model.organ_layers[organ_name].stitch_in.load_state_dict(state[organ_name]["stitch_in"])
                model.organ_layers[organ_name].stitch_out.load_state_dict(state[organ_name]["stitch_out"])
                model.organ_layers[organ_name].scale.data = state[organ_name]["scale"]
                print(f"  Loaded {organ_name} stitching")
        else:
            print(f"  WARNING: {stitching_path} not found")

    model.eval()

    # Condition 3: DUAL + Simple
    print("\n" + "-" * 70)
    print("Condition 3: DUAL Organ + Simple prompt")
    print("-" * 70)
    t0 = time.time()
    dual_simple, dual_simple_results = run_humaneval(
        model.base_model, model.tokenizer, problems, device, use_reasoning=False
    )
    t_dual_simple = time.time() - t0

    # Condition 4: DUAL + Reasoning
    print("\n" + "-" * 70)
    print("Condition 4: DUAL Organ + Reasoning prompt")
    print("-" * 70)
    t0 = time.time()
    dual_reason, dual_reason_results = run_humaneval(
        model.base_model, model.tokenizer, problems, device, use_reasoning=True
    )
    t_dual_reason = time.time() - t0

    return {
        "dual_simple": {"correct": dual_simple, "time": t_dual_simple, "results": dual_simple_results},
        "dual_reason": {"correct": dual_reason, "time": t_dual_reason, "results": dual_reason_results},
    }


def main():
    parser = argparse.ArgumentParser(description="Fair 4-condition HumanEval benchmark")
    parser.add_argument("--num-problems", type=int, default=50, help="Number of problems (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = get_device()
    num_problems = args.num_problems
    random.seed(args.seed)

    dataset = load_dataset("openai/openai_humaneval", split="test")
    indices = random.sample(range(len(dataset)), min(num_problems, len(dataset)))
    problems = [dataset[i] for i in indices]

    print("=" * 70)
    print(f"FAIR 4-CONDITION BENCHMARK: {num_problems} HumanEval Problems")
    print(f"max_new_tokens = {MAX_NEW_TOKENS} (unified)")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    total_start = time.time()

    # Run all 4 conditions
    base_results = run_base_conditions(problems, device)
    dual_results = run_dual_conditions(problems, device)

    total_time = time.time() - total_start

    # Merge results
    all_results = {**base_results, **dual_results}

    # Print summary
    n = num_problems
    bs = all_results["base_simple"]["correct"]
    br = all_results["base_reason"]["correct"]
    ds = all_results["dual_simple"]["correct"]
    dr = all_results["dual_reason"]["correct"]

    print("\n" + "=" * 70)
    print(f"RESULTS ({n} problems, max_tokens={MAX_NEW_TOKENS})")
    print("=" * 70)
    print(f"")
    print(f"{'':20s} {'Simple':>12s} {'Reasoning':>12s} {'Delta':>8s}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    print(f"{'Base Model':20s} {bs:>5d}/{n} ({100*bs/n:4.1f}%) {br:>5d}/{n} ({100*br/n:4.1f}%) {br-bs:>+5d}")
    print(f"{'DUAL Organ':20s} {ds:>5d}/{n} ({100*ds/n:4.1f}%) {dr:>5d}/{n} ({100*dr/n:4.1f}%) {dr-ds:>+5d}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    print(f"{'Delta (Organ)':20s} {ds-bs:>+5d}        {dr-br:>+5d}")
    print(f"")
    print(f"Effect decomposition:")
    print(f"  Prompt effect (Base):  {br-bs:>+d} ({100*(br-bs)/n:+.1f}%)")
    print(f"  Prompt effect (DUAL):  {dr-ds:>+d} ({100*(dr-ds)/n:+.1f}%)")
    print(f"  Organ effect (Simple): {ds-bs:>+d} ({100*(ds-bs)/n:+.1f}%)")
    print(f"  Organ effect (Reason): {dr-br:>+d} ({100*(dr-br)/n:+.1f}%)")
    print(f"  Combined effect:       {dr-bs:>+d} ({100*(dr-bs)/n:+.1f}%)")
    print(f"")
    print(f"Total time: {total_time/60:.1f} min")
    print("=" * 70)

    # Save results
    output_path = args.output or str(
        Path(__file__).resolve().parent.parent / "benchmark_results" / "fair_4condition.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "num_problems": n,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": args.seed,
            "device": device,
        },
        "results": {
            "base_simple": {"correct": bs, "total": n, "accuracy": round(100 * bs / n, 1)},
            "base_reason": {"correct": br, "total": n, "accuracy": round(100 * br / n, 1)},
            "dual_simple": {"correct": ds, "total": n, "accuracy": round(100 * ds / n, 1)},
            "dual_reason": {"correct": dr, "total": n, "accuracy": round(100 * dr / n, 1)},
        },
        "effects": {
            "prompt_effect_base": br - bs,
            "prompt_effect_dual": dr - ds,
            "organ_effect_simple": ds - bs,
            "organ_effect_reason": dr - br,
            "combined_effect": dr - bs,
        },
        "timing": {
            "base_simple_sec": round(all_results["base_simple"]["time"], 1),
            "base_reason_sec": round(all_results["base_reason"]["time"], 1),
            "dual_simple_sec": round(all_results["dual_simple"]["time"], 1),
            "dual_reason_sec": round(all_results["dual_reason"]["time"], 1),
            "total_sec": round(total_time, 1),
        },
        "per_problem": {
            "base_simple": all_results["base_simple"]["results"],
            "base_reason": all_results["base_reason"]["results"],
            "dual_simple": all_results["dual_simple"]["results"],
            "dual_reason": all_results["dual_reason"]["results"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
