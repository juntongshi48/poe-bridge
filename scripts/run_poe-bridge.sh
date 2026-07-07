#!/bin/bash
##SBATCH --account=<your-account>
##SBATCH --partition=<your-partition>
##SBATCH --qos=<your-qos>
#SBATCH --time=2-00:00:00            # Max time (days-hrs:mins:secs)
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1            # Request some number and type of GPUs
#SBATCH --job-name=poe-bridge        # Job name
#SBATCH --output=sout/%x_%j.out
#SBATCH --error=sout/%x_%j.err

set -euo pipefail
mkdir -p sout

mode="${1:-default}"
task="${2:-gsm8k}"

gpu_id=0
model_alias=dream
alg=poe-bridge
kv_window=null
max_lookahead=256
max_length=2048
verifier_size="large"
qwen_small_ckpt="Qwen/Qwen2.5-Math-1.5B-Instruct"
qwen_7b_ckpt="Qwen/Qwen2.5-Math-7B-Instruct"
limit=null
num_runs=1

# ======== PoE-Bridge Configuration ========
n_parallel_samples=4
n_low_temp_samples=1
mixture_weight=(0.3)
verify_window_size=(32)
temperature=0.2
high_temperature=0.7
anneal_temp=true
# ======== PoE-Bridge Configuration ========

output_dir="results_poe"
tag=""

# Limit MATH to 215 problem sets (~1500 questions) to reduce evaluation time and keep it comparable to other tasks.
if [ "$task" = "math" ]; then
    limit=215
fi

CMD_PREFIX=""
if [ "$mode" = "mini" ]; then
    echo "In MINI mode"
    CMD_PREFIX="python"
    output_dir="${output_dir}_mini"
    num_runs=1
    limit=50
elif [ "$mode" = "launch" ]; then
    echo "In LAUNCH mode"
    CMD_PREFIX="srun python"
    gpu_id=0
else
    echo "In DEFAULT mode"
    CMD_PREFIX="python"
fi

for mw in "${mixture_weight[@]}"; do
    for vws in "${verify_window_size[@]}"; do
        echo "  Running with: nps=${n_parallel_samples}, lts=${n_low_temp_samples}, mixture_weight=${mw}, verify_window_size=${vws}, on gpu:${gpu_id}"
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${CMD_PREFIX} \
            eval.py \
            output_dir="${output_dir}/${task}" \
            model_alias=${model_alias} \
            task=${task} \
            alg=${alg} \
            num_runs=${num_runs} \
            limit=${limit} \
            kv_window=${kv_window} \
            max_lookahead=${max_lookahead} \
            mixture_weight=${mw} \
            n_parallel_samples=${n_parallel_samples} \
            n_low_temp_samples=${n_low_temp_samples} \
            verify_window_size=${vws} \
            high_temperature=${high_temperature} \
            anneal_temp=${anneal_temp} \
            verifier_size=${verifier_size} \
            tag=${tag} \
            qwen_small_ckpt=${qwen_small_ckpt} \
            qwen_7b_ckpt=${qwen_7b_ckpt} \
            max_length=${max_length} \
            temperature=${temperature}
    done
done
