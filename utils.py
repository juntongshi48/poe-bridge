import numpy as np
from transformers import set_seed
import re

def parse_results(all_results, task_name):
    
    all_output_data = {}
    all_output_data["avg_accuracies"] = 0.0
    all_output_data["avg_throughput"] = 0.0
    all_output_data["num_tok_generated_max"] = 0
    all_output_data["avg_num_tok_generated"] = 0.0
    all_output_data["avg_num_forward_evals"] = 0.0
    all_output_data["avg_num_accepted"] = 0.0
    accuracy_ = []
    throughtput_ = []
    num_tokens_generated_ = []
    num_forward_evals_ = []
    num_tokens_generated_max_ = []

    for i, results in enumerate(all_results):
        output_data = {}
        output_data['accuracies'] = {}
        output_data['profile'] = results["profile"]
        output_data['samples'] = {}
        # Extract overall accuracies
        if 'results' in results:
            for task, task_results in results['results'].items():
                output_data['accuracies'][task] = {}
                for metric, value in task_results.items():
                    if metric.startswith('exact_match'):
                        output_data['accuracies'][task][metric] = value
                    elif metric.startswith('pass@1'):
                        output_data['accuracies'][task][metric] = value
                    elif metric.startswith('pass_at_1'):
                        output_data['accuracies'][task][metric] = value
                        
        if task_name == "math":
            accuracies = []
            for subject in output_data['accuracies']:
                if "hendrycks_math_" in subject:
                    accuracies.append(output_data["accuracies"][subject]["exact_match,flexible-extract"])
                
            output_data['accuracies']['hendrycks_math'] = {"aggregate_accuracy": np.mean(accuracies)}
        
        # Record run accuracy
        if task_name == "gsm8k":
            accuracy_.append(output_data['accuracies']['gsm8k']["exact_match,flexible-extract"])
        elif task_name == "math":
            accuracy_.append(output_data['accuracies']['hendrycks_math']["aggregate_accuracy"])
        elif task_name == "gpqa":
            accuracy_.append(output_data['accuracies']['gpqa_main_generative_n_shot']["exact_match,flexible-extract"])
        elif task_name == "humaneval":
            accuracy_.append(output_data['accuracies']['humaneval_instruct']["pass@1,create_test"])
        elif task_name == "mbpp":
            accuracy_.append(output_data['accuracies']['mbpp_instruct']["pass_at_1,extract_code"])
        # Record profilie data
        throughtput_.append(output_data['profile']['throughput_mean'])
        num_tokens_generated_max_.append(output_data['profile']['num_tokens_generated_max'])
        num_token_generated = output_data['profile']['num_tokens_generated_mean']
        num_tokens_generated_.append(num_token_generated)
        # AR does not have forward evals info, as it's simply the number of tokens generated
        if 'num_forward_evals_mean' in output_data['profile']:
            num_forward_evals_.append(output_data['profile'].get('num_forward_evals_mean', num_token_generated))

        # Extract sample data
        if 'samples' in results:

            for task, sample_list in results['samples'].items():
                output_data['samples'][task] = []
                for sample in sample_list:

                    if sample.get('filter') != 'flexible-extract' and sample.get('filter') != 'create_test' and sample.get('filter') != 'extract_code':
                        continue
                    else:
                        metric = sample.get('metrics')[0]
                    
                    is_correct = sample.get(metric, None)
                    filtered_answer = sample.get('filtered_resps', [None])[0]
                    generation = sample.get('resps', [[""]])[0][0].strip()

                    # Determine if the answer is correct by comparing the filtered response with the target
                    if task_name == "gsm8k":
                        target = sample['target']
                        gold_answer_match = re.search(r'####\s*([^\n]+)', target)
                        gold_answer = gold_answer_match.group(1).strip() if gold_answer_match else None
                        question = sample['doc']['question']
                    elif task_name == "gpqa":
                        target = sample['doc']['Correct Answer']
                        gold_answer = sample['doc']['answer']
                        question = sample['doc']['Question']
                    elif task_name == "math":
                        target = sample['doc']['solution']
                        gold_answer = sample['doc']['answer']
                        question = sample['doc']['problem']
                    elif task_name == "humaneval":
                        target = sample['doc']['canonical_solution']
                        gold_answer = sample['doc']['test']
                        question = sample['doc']['prompt']
                    elif task_name == "mbpp":
                        target = sample['doc']['code']
                        gold_answer = sample['doc']['test_list']
                        question = sample['doc']['text']


                    sample_data = {
                        'is_correct': is_correct,
                        'question': question,
                        'answer': target,
                        'ground_truth': gold_answer,
                        'filtered answer': filtered_answer,
                        'generation': generation
                    }
                    output_data['samples'][task].append(sample_data)
        all_output_data[f'run_{i}'] = output_data        
    
    # Aggregate avg statistics
    all_output_data["avg_accuracies"] = np.mean(accuracy_) if accuracy_ else None
    all_output_data["avg_throughput"] = np.mean(throughtput_)
    all_output_data["num_tok_generated_max"] = int(np.max(num_tokens_generated_max_))
    all_output_data["avg_num_tok_generated"] = np.mean(num_tokens_generated_)
    all_output_data["avg_num_forward_evals"] = np.mean(num_forward_evals_) if num_forward_evals_ else None
    all_output_data["avg_num_accepted"] = np.sum(num_tokens_generated_) / np.sum(num_forward_evals_) if num_forward_evals_ else None
       
    return all_output_data


def remove_masks(text):

    mask = "<|mask|>"
    while text.endswith(mask):
        text = text[:-len(mask)]
    return text

def drop_mask_tokens(x, masked_token_id):
    first_mask_index = (x == masked_token_id).nonzero(as_tuple=True)[0]
    if len(first_mask_index) > 0:
        return x[:first_mask_index[0]]
    return x