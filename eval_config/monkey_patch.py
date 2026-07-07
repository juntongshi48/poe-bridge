"""
Monkey patch module to apply custom task configurations to lm_eval.
"""
import os
import shutil
import site
import logging

logger = logging.getLogger(__name__)

def apply_custom_task_configs():
    """Apply custom task configurations by copying our YAML files over the original ones"""
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    lm_eval_tasks_dir = os.path.join(site_packages, "lm_eval", "tasks")
    
    # Get the directory where this script is located (eval_config)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Copy our custom YAML files to replace the originals
    custom_configs = [
        {
            "source": os.path.join(config_dir, "hendrycks_math.yaml"),
            "target": os.path.join(lm_eval_tasks_dir, "hendrycks_math", "hendrycks_math.yaml")
        },
        {
            "source": os.path.join(config_dir, "hendrycks_math_algebra.yaml"), 
            "target": os.path.join(lm_eval_tasks_dir, "hendrycks_math", "hendrycks_math_algebra.yaml")
        },
        {
            "source": os.path.join(config_dir, "gpqa_main_generative_n_shot_yaml"),
            "target": os.path.join(lm_eval_tasks_dir, "gpqa", "generative", "_gpqa_main_generative_n_shot_yaml")
        },
        {
            "source": os.path.join(config_dir, "gsm8k.yaml"),
            "target": os.path.join(lm_eval_tasks_dir, "gsm8k", "gsm8k.yaml")
        },
        {
            "source": os.path.join(config_dir, "extraction.py"),
            "target": os.path.join(site_packages, "lm_eval", "filters", "extraction.py")
        }
    ]
    
    for config in custom_configs:
        if os.path.exists(config["source"]):
            try:
                shutil.copy2(config["source"], config["target"])
                logger.info(f"Applied custom config: {os.path.basename(config['source'])}")
            except Exception as e:
                logger.warning(f"Failed to apply {os.path.basename(config['source'])}: {e}")
        else:
            logger.warning(f"Custom config not found: {config['source']}")

def adapt_from_pretrained_kwargs():
    """Monky patch the from_pretrained_kwargs that lm_eavl passed into HuggingFace models so that it works for older models like Qwen 2 series"""
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    lm_eval_tasks_dir = os.path.join(site_packages, "lm_eval", "models", "huggingface.py")
    
    
    # Get the directory where this script is located (eval_config)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Copy our custom YAML files to replace the originals
    custom_configs = [
        {
            "source": os.path.join(config_dir, "huggingface.py"),
            "target": os.path.join(site_packages, "lm_eval", "models", "huggingface.py")
        },
    ]
    
    for config in custom_configs:
        if os.path.exists(config["source"]):
            try:
                shutil.copy2(config["source"], config["target"])
                logger.info(f"Monkey Patched from_pretrained_kwargs: {os.path.basename(config['source'])}")
            except Exception as e:
                logger.warning(f"Failed to monkey patch {os.path.basename(config['source'])}: {e}")
        else:
            logger.warning(f"Monkey patch code not found: {config['source']}")