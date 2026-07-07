"""Phase 1: run naive speculative sampling vs PoE-Bridge on a shared prompt
and dump the per-step traces to demo_traces/ for Phase 2 formatting."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dream.modeling_dream import DreamModel
from utils import drop_mask_tokens


def _truncate(ids, stop_id):
    """Drop everything from the first occurrence of stop_id onward."""
    if not ids:
        return ids
    return drop_mask_tokens(torch.tensor(ids), stop_id).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="PoE-Bridge decoding trace collector")

    # Models
    parser.add_argument("--dream_ckpt", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--verifier_ckpt", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")

    # Prompt / generation
    parser.add_argument("--prompt", type=str, default="Alice has 14 stickers. She gives 5 to Bob and then buys 8 more. How many stickers does Alice have now?")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)

    # PoE-Bridge hyperparameters (match exp/run_poe_gsm8k.sh defaults)
    parser.add_argument("--max_lookahead", type=int, default=256)
    parser.add_argument("--kv_window", type=int, default=None)
    parser.add_argument("--mixture_weight", type=float, default=0.7)
    parser.add_argument("--n_parallel_samples", type=int, default=3)
    parser.add_argument("--n_low_temp_samples", type=int, default=1)
    parser.add_argument("--verify_window_size", type=int, default=32)
    parser.add_argument("--high_temperature", type=float, default=0.7)
    parser.add_argument("--anneal_temp", action="store_true", default=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="demo_traces")

    return parser.parse_args()


def run_one(model, verifier_model, tokenizer, input_ids, args, *,
            n_parallel_samples, mixture_weight, label):
    """Run diffusion_generate once, collect the trace, return everything as a dict."""
    trace = []

    def trace_hook(step_idx, trace_dict):
        trace.append(trace_dict)

    start = time.time()
    out = model.diffusion_generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        temperature=args.temperature,
        top_p=args.top_p,
        alg="poe-bridge",
        verifier_model=verifier_model,
        kv_window=args.kv_window,
        max_lookahead=args.max_lookahead,
        mixture_weight=mixture_weight,
        n_parallel_samples=n_parallel_samples,
        n_low_temp_samples=min(args.n_low_temp_samples, n_parallel_samples),
        verify_window_size=args.verify_window_size,
        high_temperature=args.high_temperature,
        anneal_temp=args.anneal_temp,
        generation_trace_hook_func=trace_hook,
    )
    elapsed = time.time() - start

    full_sequence = out.sequences[0].tolist()
    prompt_len = input_ids.shape[1]
    generated_ids = full_sequence[prompt_len:]
    final_text = tokenizer.decode(
        out.sequences[0], skip_special_tokens=True,
    )

    profile = getattr(out, "profile", None)

    # Truncate trailing padding: poe_bridge_sample replaces remaining masks with EOS at end.
    eos_id = tokenizer.eos_token_id
    full_sequence = _truncate(full_sequence, eos_id)
    generated_ids = _truncate(generated_ids, eos_id)
    for step in trace:
        for cand in step["candidates"]:
            cand["dlm_draft_tokens"] = _truncate(cand["dlm_draft_tokens"], eos_id)
            cand["accepted_tokens"] = _truncate(cand["accepted_tokens"], eos_id)

    acceptance_counts = getattr(profile, "acceptance_counts", None)
    num_accepted_mean = np.mean(acceptance_counts) if acceptance_counts is not None else None
    return {
        "label": label,
        "config": {
            "n_parallel_samples": n_parallel_samples,
            "mixture_weight": mixture_weight,
            "max_lookahead": args.max_lookahead,
            "kv_window": args.kv_window,
            "n_low_temp_samples": min(args.n_low_temp_samples, n_parallel_samples),
            "verify_window_size": args.verify_window_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "high_temperature": args.high_temperature,
            "anneal_temp": args.anneal_temp,
        },
        "tokenizer_name": args.dream_ckpt,
        "prompt_text": args.prompt,
        "prompt_token_ids": input_ids[0].tolist(),
        "final_sequence_ids": full_sequence,
        "generated_ids": generated_ids,
        "final_text": final_text,
        "elapsed_sec": elapsed,
        "num_forward_evals": getattr(profile, "num_forward_evals", None),
        "num_tokens_generated": getattr(profile, "num_tokens_generated", None),
        "num_accepted_mean": num_accepted_mean,
        "trace": trace,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dream model: {args.dream_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.dream_ckpt, trust_remote_code=True,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    print(f"Loading verifier model: {args.verifier_ckpt}")
    verifier_model = AutoModelForCausalLM.from_pretrained(
        args.verifier_ckpt, trust_remote_code=True,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    ).to(model.device)

    runs = [
        dict(label="naive_speculative", n_parallel_samples=1, mixture_weight=0.0),
        dict(label="poe_bridge",
             n_parallel_samples=args.n_parallel_samples,
             mixture_weight=args.mixture_weight),
    ]

    for r in runs:
        print(f"\n=== Running {r['label']} (K={r['n_parallel_samples']}, mw={r['mixture_weight']}) ===")
        # Re-seed before each run so both methods see the same initial randomness.
        set_seed(args.seed)
        result = run_one(
            model, verifier_model, tokenizer, input_ids, args,
            n_parallel_samples=r["n_parallel_samples"],
            mixture_weight=r["mixture_weight"],
            label=r["label"],
        )
        out_path = os.path.join(args.output_dir, f"{r['label']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved trace -> {out_path}")
        print(f"  steps: {len(result['trace'])}  tokens: {result['num_tokens_generated']}  "
              f"time: {result['elapsed_sec']:.2f}s")
        print(f"  final: {result['final_text'][:200]}{'...' if len(result['final_text']) > 200 else ''}")


if __name__ == "__main__":
    main()
