from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import torch
import wandb
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info
import json, os
from PIL import Image



text_qv_modules = [f"layers.{i}.self_attn.q_proj" for i in range(36)] + [f"layers.{i}.self_attn.v_proj" for i in range(36)]

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = Qwen2_5_VLProcessor.from_pretrained(model_id)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=text_qv_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)


# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Freeze visual encoder
for name, param in model.named_parameters():
    if name.startswith("visual"):
        param.requires_grad = False

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

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

from tqdm import tqdm
import sys

def format_data(sample):
    """
    Format a single dataset sample into the required structure.
    """
    base_dir = "/content/drive/MyDrive/MMHRC/gqa/images"
    image_path = os.path.join(base_dir, sample["image"])
    image=Image.open(image_path).convert("RGB")

    # print(sample)
    output_text = (
      f"<thought> {sample['thought'].strip()}</thought>\n<answer>{sample['answer']}</answer>"
    )
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{sample['question']}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}],
        },
    ]



class SFT_Dataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = image_dir
        
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.samples.append(ex)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        item = self.samples[idx]

        image_path = os.path.join(self.image_dir, item['image'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return self.__getitem__((idx + 1) % len(self.samples))

        return item

base_dir = "/content/drive/MyDrive/MMHRC/gqa"
train_dataset=SFT_Dataset(os.path.join(base_dir, "gqa_cot_train_1k.jsonl"),os.path.join(base_dir, "images"))
eval_dataset=SFT_Dataset(os.path.join(base_dir, "gqa_cot_val_1k.jsonl"),os.path.join(base_dir, "images"))
train_dataset = [format_data(sample) for sample in tqdm(train_dataset, desc="Formatting train set", file=sys.stdout)]
eval_dataset = [format_data(sample) for sample in tqdm(eval_dataset, desc="Formatting eval set", file=sys.stdout)]

def collate_fn(examples):
    """
    Data collator to prepare a batch of examples.

    This function applies the chat template to texts, processes the images,
    tokenizes the inputs, and creates labels with proper masking.
    """
    # Apply chat template to each example (no tokenization here)
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    # Process visual inputs for each example
    image_inputs = [process_vision_info(example)[0] for example in examples]

    # Tokenize texts and images into tensors with padding
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # Create labels by cloning input_ids and mask the pad tokens
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Determine image token IDs to mask in the labels (model specific)
    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    
    return batch

from trl import SFTConfig

# Configure training arguments
training_args = SFTConfig(
    output_dir="./qwen-sft-052814",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    learning_rate=2e-4,  # Learning rate for training
    logging_steps=1,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    load_best_model_at_end=True,  # Load the best model after training
    bf16=True,  # Use bfloat16 precision
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    report_to="wandb",  # Reporting tool for tracking metrics
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns = False
)


import wandb

wandb.init(
    project="qwen2_5-vl-sft",  # change this
    name="qwen2_5-vl-sft",  # change this
    config=training_args,
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=lora_config,
    processing_class=processor.tokenizer,
)

trainer.train()

# trainer.save_model(training_args.output_dir)
# trainer.save_processor(training_args.output_dir)
model.save_pretrained("./qwen_vl_sft_052814")
processor.save_pretrained("./qwen_vl_sft_052814")
trainer.save_model(training_args.output_dir)
