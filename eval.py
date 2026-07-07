import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import argparse
import json
import logging
import torch
from lm_eval import evaluator
from harness import DreamEvalHarness, ProfileEvalHarness
from utils import parse_results
from transformers import AutoTokenizer
from dream.modeling_dream import DreamModel

import hydra

from collections import Counter
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply custom task configurations
from eval_config.monkey_patch import apply_custom_task_configs, adapt_from_pretrained_kwargs
apply_custom_task_configs()

# Define the tasks you want to support
TASKS = {
    'gsm8k': 'gsm8k', 
    'math': 'hendrycks_math', 
    'gpqa': 'gpqa_main_generative_n_shot', 
    'humaneval': 'humaneval_instruct',
    'mbpp': 'mbpp_instruct',
} # Use the exact task names from lm-eval task registry


def get_model(args):
    
    model_alias = args.model_alias
    alg = args.alg

    max_lookahead = args.max_lookahead
    kv_window = args.kv_window
    mixture_weight = args.mixture_weight
    n_parallel_samples = args.n_parallel_samples
    n_low_temp_samples = args.n_low_temp_samples
    verify_window_size = args.verify_window_size
    temperature = args.temperature
    top_p = args.top_p
    high_temperature = args.high_temperature
    anneal_temp = args.anneal_temp
    num_steps = args.num_steps
    task_name = args.task

    logger.info(f"Configuring model details for alias: {model_alias}")
    if model_alias == "qwen7b":
        if args.qwen_7b_ckpt is None:
            raise ValueError("--qwen_7b_ckpt is required when --model_alias=qwen7b")
        if task_name in ["humaneval", "mbpp"]:
            adapt_from_pretrained_kwargs()
        model = ProfileEvalHarness(pretrained=args.qwen_7b_ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", max_length=16384)
    elif model_alias == "qwen_small":
        if args.qwen_small_ckpt is None:
            raise ValueError("--qwen_small_ckpt is required when --model_alias=qwen_small")
        model = ProfileEvalHarness(pretrained=args.qwen_small_ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", max_length=16384)
    elif model_alias == "dream":
        if args.dream_ckpt is None:
            raise ValueError("--dream_ckpt is required when --model_alias=dream")
        if args.alg == "poe-bridge" and args.qwen_small_ckpt is None:
            raise ValueError("--qwen_small_ckpt is required when --model_alias=dream and --alg=poe-bridge")
        dream = DreamModel.from_pretrained(args.dream_ckpt, 
                                               trust_remote_code=True,  
                                               attn_implementation="sdpa", 
                                               torch_dtype=torch.bfloat16, 
                                               device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
        verifier_ckpt = None
        if args.verifier_size =='small':
            verifier_ckpt = args.qwen_small_ckpt
        elif args.verifier_size == 'large':
            verifier_ckpt = args.qwen_7b_ckpt
        max_gen_toks = 512 if task_name=="math" else 256    # the default DLM context window
        if args.max_length is not None:
            max_gen_toks = args.max_length
        model = DreamEvalHarness(
            pretrained=dream,
            tokenizer=tokenizer,
            alg=alg,
            max_lookahead=max_lookahead,
            kv_window=kv_window,
            mixture_weight=mixture_weight,
            n_parallel_samples=n_parallel_samples,
            n_low_temp_samples=n_low_temp_samples,
            verify_window_size=verify_window_size,
            temperature=temperature,
            top_p=top_p,
            high_temperature=high_temperature,
            anneal_temp=anneal_temp,
            num_steps=num_steps,
            max_gen_toks=max_gen_toks,
            verifier_ckpt=verifier_ckpt,
        )
    else:
        raise ValueError(f"Unknown model alias: {model_alias}. Must be one of 'qwen7b', 'qwen_small', 'dream'.")

    return model

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args):
    # Validate algorithm choices based on model alias
    valid_algs = {
        "dream": ["leftright", "poe-bridge", "entropy", "origin"],
        "qwen7b": ["leftright", None],
        "qwen_small": ["leftright", None]
    }
    
    if args.model_alias in valid_algs:
        if args.alg not in valid_algs[args.model_alias]:
            valid_alg_str = ", ".join([str(alg) for alg in valid_algs[args.model_alias]])
            raise ValueError(f"Invalid algorithm '{args.alg}' for model '{args.model_alias}'. "
                           f"Valid algorithms for {args.model_alias}: {valid_alg_str}")
    
    model = get_model(args)
    
    ar_model_size = ""
    if args.model_alias == "dream":
        ar_model_size = args.verifier_size
    elif args.model_alias == "qwen7b":
        ar_model_size = "large"
    elif args.model_alias == "qwen_small":
        ar_model_size = "small"
    
    ar_model_tag = None
    if ar_model_size == "large":
        ar_model_tag = args.qwen_7b_ckpt.split("/Qwen")[1]
        ar_model_tag = "-".join(ar_model_tag.split("-")[1:])
    elif ar_model_size == "small":
        ar_model_tag = args.qwen_small_ckpt.split("/Qwen")[1]
        ar_model_tag = "-".join(ar_model_tag.split("-")[1:])
        
    output_dir = os.path.join(f"{args.output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # max_length is defined only for generating with AR models
    # it specifies the max length of (input + generated) tokens
    max_length = 16384
    if args.model_alias == "qwen7b" and 'Math' in args.qwen_7b_ckpt:
        max_length = 4096
    elif args.model_alias == "qwen_small" and 'Math' in args.qwen_small_ckpt:
        max_length = 4096
    
    task_str = ""
    if args.alg is not None:
        task_str += f"_{args.alg}"

    if args.max_lookahead is not None:
        task_str += f"_M={args.max_lookahead}"
    if args.kv_window is not None:
        task_str += f"_W={args.kv_window}"
    if args.mixture_weight is not None:
        task_str += f"_w={args.mixture_weight}"
    if args.n_parallel_samples is not None:
        task_str += f"_N={args.n_parallel_samples}"
    if args.n_low_temp_samples is not None:
        task_str += f"_LTS={args.n_low_temp_samples}"
    if args.verify_window_size is not None:
        if args.max_lookahead is not None:
            assert args.verify_window_size <= args.max_lookahead, "verify_window_size should be less than or equal to max_lookahead"
        task_str += f"_V={args.verify_window_size}"
    if args.num_steps is not None:
        task_str += f"_num_steps={args.num_steps}"
    if args.verifier_size is not None and args.alg=='poe-bridge':
        task_str += f"_ar-{args.verifier_size}"
    if args.temperature is not None:
        task_str += f"_temp={args.temperature}"
    if args.top_p is not None:
        task_str += f"_topp={args.top_p}"
    if args.max_length is not None:
        task_str += f"_genlen={args.max_length}"
        max_length = args.max_length
    if args.num_runs >= 1:
        task_str += f"_runs={args.num_runs}"
    if ar_model_tag is not None:
        task_str += f"_{ar_model_tag}"
    if args.tag:
        task_str += f"_{args.tag}"
    output_filename = f"{args.model_alias}{task_str}_limit={args.limit}.json"
    output_path = os.path.join(output_dir, output_filename)
    logger.info(f"Results will be saved to: {output_path}")
    
    
    task_name = args.task
    
    if task_name == "math":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: boxed{ANSWER}."
    elif task_name == "gpqa":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: (LETTER)."
    elif task_name == "gsm8k":
        system_instruction = "You are a helpful assistant. Conclude by presenting the final answer as an integer in the format: boxed{ANSWER}."
    else:
        system_instruction = "You are a helpful assistant."
        
    if "qwen" in args.model_alias:
        system_instruction = "You are a helpful assistant." # Normal prompt for qwen models

    task = [TASKS[task_name]]
    
    gen_kwargs = None
    if "qwen" in args.model_alias:
        gen_kwargs = ""
        do_sample = False
        if args.temperature is not None:
            gen_kwargs += f"temperature={args.temperature},"
            do_sample = True
        if args.top_p is not None:
            gen_kwargs += f"top_p={args.top_p},"
            do_sample = True
        gen_kwargs += f"do_sample={do_sample},"
        gen_kwargs += f"max_length={max_length},"
        print(f"max_length is set to {max_length} for generation.")
    
    all_results = []
    for i in range(args.num_runs):
        logger.info(f"Starting run {i+2} / {args.num_runs}")
        seed_configs = {}
        if i > 0:
            seed_configs = {
                "random_seed": 42 + i,
                "numpy_random_seed": 42 + i,
                "torch_random_seed": 42 + i,
            }
        results = evaluator.simple_evaluate(
            model=model,
            tasks=task,
            batch_size=1,
            limit=args.limit,
            log_samples=True,    
            write_out=True,    
            num_fewshot=0, 
            apply_chat_template=True,
            system_instruction=system_instruction,
            gen_kwargs=gen_kwargs,
            confirm_run_unsafe_code=True,
            **seed_configs
        )
        results["profile"] = model.get_profile()
        if "num_accepted" in results["profile"]:
            num_accepted = results["profile"]["num_accepted"]
            results["profile"].pop("num_accepted")
            save_path = output_path.replace(".json", "_accept_hist.png")
            plot_accept_counts(num_accepted, save_path)
        all_results.append(results)
    parsed_results = parse_results(all_results, task_name=task_name)
    
    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)
    
def plot_accept_counts(num_accepted, save_path):
    counts = Counter(num_accepted)
    total = len(num_accepted)

    xs = sorted(counts)
    ys = [counts[x] / total * 100 for x in xs]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Num Accepted")
    plt.ylabel("Percentage (%)")
    plt.title("Histogram of Num Accepted (%)")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
    
    