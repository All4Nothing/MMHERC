from grpotrainer import Qwen2VLGRPOTrainer
from rewards import semantic_reward_func, format_reward

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


lora_model_path = "/content/drive/MyDrive/MMHRC/gqa/qwen_vl_sft_0527"

peft_config = PeftConfig.from_pretrained(lora_model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, lora_model_path)

model.print_trainable_parameters()

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


from trl import TrlParser, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def get_checkpoint(training_args):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


class CustomDataset:
    def __init__(self, list_data_dict=None, script_args=None, processor=None, json_data_path=None):
        super(CustomDataset, self).__init__()
        self.script_args = script_args
        self.processor = processor

        # Load data from JSON file if path is provided
        if json_data_path and os.path.exists(json_data_path):
            import json

            with open(json_data_path, "r") as f:
              if json_data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
              else:
                data = json.load(f)
            self.list_data_dict = data
        else:
            self.list_data_dict = list_data_dict or []

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example["image"])
            image = Image.open(image_path).convert("RGB")

            # Resize image if needed to meet min/max size requirements
            w, h = image.size

            if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio for small images
                if w < h:
                    new_w = 28
                    new_h = int(h * (28 / w))
                else:
                    new_h = 28
                    new_w = int(w * (28 / h))
                image = image.resize((new_w, new_h), Image.LANCZOS)
            elif w > 512 or h > 512:
                # Calculate new dimensions maintaining aspect ratio for large images
                if w > h:
                    new_w = 512
                    new_h = int(h * (512 / w))
                else:
                    new_h = 512
                    new_w = int(w * (512 / h))
                image = image.resize((new_w, new_h), Image.LANCZOS)
            else:
                # Image is within acceptable dimensions, no resize needed
                new_w, new_h = w, h
        else:
            image = None

        return {
            "image": image,
            "question": example["question"],
            "answer": example["answer"],
            "prompt": (
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{example['question']}\n\nYou must follow this EXACT format:\n<thought>your reasoning</thought>\n<answer>one word</answer>\n\nRules:\n1. You must use both <thought> and <answer> tags\n2. Your answer must be exactly ONE WORD\n3. Do not add any text outside these tags\n4. Do not use any other tags or formats"},
                    ],
                },
            ),
        } 


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
    # 1. 모델 설정
    model_args = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="float16",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # 2. 사용자 설정
    class ScriptArguments:
        json_data_path = "/content/drive/MyDrive/MMHRC/gqa/gqa_cot_train_2K_1.jsonl"
        image_root = "/content/drive/MyDrive/MMHRC/gqa/images"
        max_pixels = 12845056
        min_pixels = 3136
        tokenizer_name_or_path = None

    script_args = ScriptArguments()

    # 3. 학습 설정
    training_args = GRPOConfig(
        output_dir="./grpo_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
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
        logging_steps=1,
        seed=42,
        report_to=["wandb"],
        run_name="qwen2.5-grpo"
    )

    # 4. 모델 로드
    lora_model_path = "/content/drive/MyDrive/MMHRC/gqa/qwen_vl_sft_0527"
    peft_config = PeftConfig.from_pretrained(lora_model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    # 5. GRPO 학습 실행
    grpo_function(model, model_args, script_args, training_args)