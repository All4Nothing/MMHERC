from grpotrainer import Qwen2VLGRPOTrainer
from grpo.rewards import semantic_reward_func, format_reward

import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer, AutoTokenizer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import torch.nn.functional as F
from trl import GRPOTrainer, GRPOConfig, ModelConfig
from peft import PeftModel

from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig


from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoProcessor,
    BitsAndBytesConfig,
)

import logging
import os
from dataclasses import dataclass
from datetime import datetime
import random
import re
import torch
import yaml

from transformers.trainer_utils import get_last_checkpoint

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

import datasets
from datasets import load_dataset
from torch.utils.data import Dataset

import sys
from typing import Optional, Tuple

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    # flash_attn_varlen_func,
)
import torch
from typing import Tuple

from model import load_model
from grpo.grpo_dataset import CustomDataset

from trl import TrlParser, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from PIL import Image




lora_model_path = "./checkpoints"
model, processor = load_model(lora_model_path)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

system_message = """You are a Vision Language Model that specializes in interpreting general visual scenes, such as indoor spaces, objects, people, or environments.

Your task is to look at the provided image and answer natural language questions about it.

You must follow this EXACT format:
<thought>your reasoning</thought>
<answer>your answer(one or two words)</answer>

Rules:
1. You must use both <thought> and <answer> tags
2. Your answer must be exactly ONE or TWO WORDS only
3. Do not add any text outside these tags
4. Do not use any other tags or formats
5. Be concise and accurate based on the visual evidence

Avoid unnecessary explanation. Focus only on answering the question based on the visual information in the image."""


def get_checkpoint(training_args):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def grpo_function(model, model_args, script_args, training_args):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    model_name = model_args.model_name_or_path
    tokenizer_path = script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_name

    # Load the appropriate model and processor based on model name
    if "Qwen2.5-VL" in model_name:
        processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )
    else:  # Default to Qwen2-VL
        processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )

    # Create CustomDataset instances
    train_dataset = CustomDataset(
        script_args=script_args,
        processor=processor,
        json_data_path=script_args.json_data_path,
    )
    print(f"Created datasets with {len(train_dataset)} training examples")
    print(f"Sample example: {train_dataset[0]}")


    print(f"Created training dataset with {len(train_dataset)} examples")
    print(f"Sample training example: {train_dataset[0]}")

    # Choose your reward functions
    # chosen_reward_funcs = [semantic_reward_func, format_reward]
    chosen_reward_funcs = [semantic_reward_func, format_reward]

    trainer = Qwen2VLGRPOTrainer(
        model=model,
        reward_funcs=chosen_reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        # peft_config=get_peft_config(model_args),
        peft_config=None,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming from checkpoint at {last_checkpoint}.")

    logger.info("*** Starting training ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


    logger.info("*** Done ***")


if __name__ == "__main__":
     # from trl import ModelConfig, get_peft_config

    # 1. 모델 설정
    model_args = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="float16",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # 2. 사용자 설정
    class ScriptArguments:
        json_data_path = "/content/drive/MyDrive/MMHRC/gqa/gqa_cot_train_1k.jsonl"
        image_root = "/content/drive/MyDrive/MMHRC/gqa/images"
        max_pixels = 12845056
        min_pixels = 3136
        tokenizer_name_or_path = None

    script_args = ScriptArguments()

    # 3. 학습 설정
    training_args = GRPOConfig(
        output_dir="./grpo_checkpoints",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        beta=0.04,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_prompt_length=1024,
        max_completion_length=128,
        num_generations=4,
        use_vllm=False,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        seed=42,
        report_to=["wandb"],
        run_name="qwen2.5-grpo"
    )

    model, processor = load_model(lora_model_path)

    grpo_function(model, model_args, script_args, training_args)