from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import stop_sequences_criteria
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import time
import numpy as np


class ProfileEvalHarness(HFLM):
    def __init__(self, **args):
        super().__init__(**args)
        self.profile = {}
        
    @torch.inference_mode()
    def manual_generate(self, input_ids, stop, pad_token_id, generation_kwargs):
        '''
        x: (b, l)
        block_length:
        return: (b, block_size) of normalized probabilities
        '''
        max_length = generation_kwargs.get("max_new_tokens", 10000)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        past_key_values = None
        for i in range(max_length):
            if past_key_values is None:
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            else:
                outputs = self.model(
                    input_ids=next_token,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:,-1:,:] # (b, 1, |V|)
            next_token = torch.argmax(next_token_logits, dim=-1) # (b, 1)
            
            next_token[finished, :] = pad_token_id
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            for stop_token in stop:
                stop_token = torch.tensor(stop_token, device=input_ids.device).unsqueeze(0) #(1, len_stop)
                new_finished = input_ids[:, -stop_token.shape[1]:].eq(stop_token).all(dim=-1)
                finished = finished | new_finished
            
            if finished.all():
                break
        return input_ids
        
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        
        start = time.time()
        
        # stop_tokens = self.tokenizer(stop)['input_ids']
        # output = self.manual_generate(context, stop_tokens, self.tokenizer.pad_token_id, generation_kwargs)
        
        output = self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
        stop_time = time.time()
        
        
        self.log_profile({
            "num_tokens_generated": output.shape[1] - context.shape[1],
            "total_time": stop_time - start,
        })
        
        
        
        return output
    
    def log_profile(self, profile):
        
        for k, v in profile.items():
            if k not in self.profile:
                self.profile[k] = []
            self.profile[k].append(v)
            
    def get_profile(self):
        num_tokens_generated = np.array(self.profile["num_tokens_generated"])
        total_times = np.array(self.profile["total_time"])
        throughputs = num_tokens_generated / total_times
        
        throughput_mean = throughputs.mean()
        throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
        num_tokens_generated_max = int(np.max(num_tokens_generated))
        num_tokens_generated_mean = num_tokens_generated.mean()
        num_tokens_generated_stderr = num_tokens_generated.std(ddof=1) / math.sqrt(len(num_tokens_generated))
        total_time_mean = total_times.mean()
        total_time_stderr = total_times.std(ddof=1) / math.sqrt(len(total_times))
        
        result = {"throughput_mean": throughput_mean,
                    "throughput_stderr": throughput_stderr,
                    "total_time_mean": total_time_mean,
                    "total_time_stderr": total_time_stderr,
                    "num_tokens_generated_max": num_tokens_generated_max,
                    "num_tokens_generated_mean": num_tokens_generated_mean,
                    "num_tokens_generated_stderr": num_tokens_generated_stderr}

        return result


class DreamEvalHarness(HFLM):
    def __init__(self, pretrained, tokenizer, **args):
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, **args)
        self.model_alias = "dream"
        self.alg = args.get("alg", "origin")
        self.max_lookahead = args.get("max_lookahead", None)
        self.kv_window = args.get("kv_window", None)
        self.mixture_weight = args.get("mixture_weight", None)
        self.n_parallel_samples = args.get("n_parallel_samples", None)
        self.n_low_temp_samples = args.get("n_low_temp_samples", None)
        self.verify_window_size = args.get("verify_window_size", None)
        self.num_steps = args.get("num_steps", None)
        
        self.temperature = args.get("temperature", None)
        self.top_p = args.get("top_p", None)
        self.high_temperature = args.get("high_temperature", None)
        self.anneal_temp = args.get("anneal_temp", None)
        
        if self.num_steps is None:
            self.num_steps = 256
        
        self.max_gen_toks_value = args.get("max_gen_toks", 256)
        
        if self.alg == "poe-bridge":
            verifier_ckpt = args.get("verifier_ckpt", None)
            if verifier_ckpt is None:
                raise ValueError("--qwen_ckpt (verifier_ckpt) is required when alg=poe-bridge")
            self.verifier_model = AutoModelForCausalLM.from_pretrained(
                verifier_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
            )
        
        self.profile = {}
            
        
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError
    
    def log_profile(self, profile):
        
        for k, v in profile.items():
            if k not in self.profile:
                if type(v) == dict:
                    self.profile[k] = {}
                    for sub_k, sub_v in v.items():
                        self.profile[k][sub_k] = []
                else:
                    self.profile[k] = []
            if type(v) == dict:
                for sub_k, sub_v in v.items():
                    self.profile[k][sub_k].append(sub_v)
            elif type(v) == list:
                self.profile[k].extend(v)
            else:
                self.profile[k].append(v)
            
    
    def get_profile(self):
        
        if self.alg != "leftright" and self.alg != "poe-bridge":
            num_tokens_generated = np.array(self.profile["num_tokens_generated"])
            total_times = np.array(self.profile["total_time"])
            num_forward_evals = np.array(self.profile["num_forward_evals"])
            num_accepted = np.array(self.profile["acceptance_counts"])
            
            
            throughputs = num_tokens_generated / total_times
            throughput_mean = throughputs.mean()
            throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
            
            total_time_mean = np.mean(total_times)
            total_time_stderr = np.std(total_times, ddof=1) / math.sqrt(len(total_times))
        
            num_tokens_generated_mean = np.mean(num_tokens_generated)
            num_tokens_generated_stderr = np.std(num_tokens_generated, ddof=1) / math.sqrt(len(num_tokens_generated))
            num_tokens_generated_max = int(np.max(num_tokens_generated))
            
            num_foward_evals_mean = np.mean(num_forward_evals)
            num_foward_evals_stderr = np.std(num_forward_evals, ddof=1) / math.sqrt(len(num_forward_evals))
            num_accepted_mean = np.mean(num_accepted)
            num_accepted_stderr = np.std(num_accepted, ddof=1) / math.sqrt(len(num_accepted))
            num_accepted_max = int(max(num_accepted))
            
            result = {"throughput_mean": throughput_mean,
                    "throughput_stderr": throughput_stderr,
                    "total_time_mean": total_time_mean,
                    "total_time_stderr": total_time_stderr,
                    "num_tokens_generated_mean": num_tokens_generated_mean,
                    "num_tokens_generated_stderr": num_tokens_generated_stderr,
                    "num_tokens_generated_max": num_tokens_generated_max,
                    "num_forward_evals_mean": num_foward_evals_mean,
                    "num_forward_evals_stderr": num_foward_evals_stderr,
                    "num_accepted_mean": num_accepted_mean,
                    "num_accepted_stderr": num_accepted_stderr,
                    "num_accepted_max": num_accepted_max,
                    }
            
            return result
                    
        num_forward_evals = np.array(self.profile["num_forward_evals"])
        num_tokens_generated = np.array(self.profile["num_tokens_generated"])
        total_times = np.array(self.profile["total_time"])
        num_accepted = np.array(self.profile["acceptance_counts"])
        
        
        num_foward_evals_mean = np.mean(num_forward_evals)
        num_foward_evals_stderr = np.std(num_forward_evals, ddof=1) / math.sqrt(len(num_forward_evals))
        num_tokens_generated_mean = np.mean(num_tokens_generated)
        num_tokens_generated_stderr = np.std(num_tokens_generated, ddof=1) / math.sqrt(len(num_tokens_generated))
        num_tokens_generated_max = int(np.max(num_tokens_generated))
        total_time_mean = np.mean(total_times)
        total_time_stderr = np.std(total_times, ddof=1) / math.sqrt(len(total_times))
        num_accepted_mean = np.mean(num_accepted)
        num_accepted_stderr = np.std(num_accepted, ddof=1) / math.sqrt(len(num_accepted))
        num_accepted_max = int(max(num_accepted))
        frac_all_rejected = (num_accepted==1).mean()

        throughputs = num_tokens_generated / total_times
        throughput_mean = throughputs.mean()
        throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
        
        result = {
            "throughput_mean": throughput_mean,
            "throughput_stderr": throughput_stderr,
            "total_time_mean": total_time_mean,
            "total_time_stderr": total_time_stderr,
            "num_tokens_generated_mean": num_tokens_generated_mean,
            "num_tokens_generated_stderr": num_tokens_generated_stderr,
            "num_tokens_generated_max": num_tokens_generated_max,
            "num_forward_evals_mean": num_foward_evals_mean,
            "num_forward_evals_stderr": num_foward_evals_stderr,
            "num_accepted_mean": num_accepted_mean,
            "num_accepted_stderr": num_accepted_stderr,
            "num_accepted_max": num_accepted_max,
            "frac_all_rejected": frac_all_rejected,
            "num_accepted": num_accepted.tolist(),
        }
        
        if "detailed_time" in self.profile:
            detailed_time = self.profile["detailed_time"]
            result["detailed_time"] = {}
            for k, v in detailed_time.items():
                times = np.array(v)
                time_mean = np.mean(times)
                time_stderr = np.std(times, ddof=1) / math.sqrt(len(times))
                result["detailed_time"][f"{k}_mean"] = time_mean
                # result["detailed_time"][f"{k}_stderr"] = time_stderr
        
        return result
    
    @property
    def max_gen_toks(self) -> int:
        return self.max_gen_toks_value
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):

        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
            
        if self.alg == "poe-bridge":
            outputs = self.model.diffusion_generate(
                context,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                steps=self.num_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                alg="poe-bridge",
                alg_temp=0.,
                max_lookahead=self.max_lookahead,
                kv_window=self.kv_window,
                mixture_weight=self.mixture_weight,
                n_parallel_samples=self.n_parallel_samples,
                n_low_temp_samples=self.n_low_temp_samples,
                verify_window_size=self.verify_window_size,
                high_temperature = self.high_temperature,
                anneal_temp = self.anneal_temp,
                verifier_model=self.verifier_model,
                return_dict_in_generate=True,
                )
            
            self.log_profile(outputs.profile)
            
            return outputs.sequences
        
        elif self.alg == "leftright":
            
            outputs = self.model.diffusion_generate(
                context,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                steps=self.num_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                alg="leftright",
                alg_temp=0.,
                max_lookahead=self.max_lookahead,
                kv_window=self.kv_window,
                mixture_weight=self.mixture_weight,
                n_parallel_samples=self.n_parallen_samples,
                n_low_temp_samples=self.n_low_temp_samples,
                verify_window_size=self.verify_window_size,
                high_temperature = self.high_temperature,
                anneal_temp = self.anneal_temp,
                return_dict_in_generate=True,
                )
            
            self.log_profile(outputs.profile)
            
            return outputs.sequences
        
        start = time.time()
        outputs = self.model.diffusion_generate(
            context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            steps=self.num_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            alg=self.alg,
            alg_temp=0.,
            max_lookahead=self.max_lookahead,
        )
        end = time.time()
        
        sequences = outputs.sequences
        num_tokens_generated = -context.shape[-1]
        for value in sequences[0]:
            if value == self.tokenizer.eos_token_id:
                break
            num_tokens_generated+=1
        
        profile = outputs.profile
        self.log_profile(profile)
          
        new_profile_entries =  {"num_tokens_generated": num_tokens_generated,
                   "total_time": end - start}
        
        self.log_profile(new_profile_entries)
        return sequences

