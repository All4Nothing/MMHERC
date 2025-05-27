import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

class SFT_Dataset(Dataset):
    def __init__(self, jsonl_path, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []
        
        with open(jsonl_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.samples.append(ex)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
            # Resize image if needed
            w, h = image.size
            if w < 28 or h < 28:
                if w < h:
                    new_w = 28
                    new_h = int(h * (28 / w))
                else:
                    new_h = 28
                    new_w = int(w * (28 / h))
                image = image.resize((new_w, new_h), Image.LANCZOS)
            elif w > 512 or h > 512:
                if w > h:
                    new_w = 512
                    new_h = int(h * (512 / w))
                else:
                    new_h = 512
                    new_w = int(w * (512 / h))
                image = image.resize((new_w, new_h), Image.LANCZOS)
        except Exception as e:
            print(f"image error: {image_path} - {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Format the output text with thought and answer
        output_text = (
            f"<thought>{item['thought'].strip()}</thought>\n"
            f"<answer>{item['answer'].strip()}</answer>"
        )

        # Create the input message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{item['question']}\n\nYou must follow this EXACT format:\n<thought>your reasoning</thought>\n<answer>one word</answer>\n\nRules:\n1. You must use both <thought> and <answer> tags\n2. Your answer must be exactly ONE WORD\n3. Do not add any text outside these tags\n4. Do not use any other tags or formats"},
                ]
            },
        ]

        # Process the input
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process image and text inputs
        model_inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        )

        # Process the target text
        with self.processor.tokenizer.as_target_tokenizer():
            target = self.processor.tokenizer(
                output_text + "<|im_end|>",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1024
            )

        # Prepare the final inputs
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        model_inputs["labels"] = target["input_ids"].squeeze(0)

        return model_inputs

def train_sft():
    # Initialize wandb
    wandb.init(
        project="mmhrc",
        name="sft_qwen_3k_1",
        config={
            "model": "Qwen2.5-VL-3B-Instruct",
            "batch_size": 2,
            "grad_accum_steps": 4,
            "learning_rate": 2e-5,
        }
    )

    # Load model and processor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        attn_implementation="flash_attention_2"
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Freeze visual encoder
    for name, param in model.named_parameters():
        if name.startswith("visual"):
            param.requires_grad = False

    # Configure LoRA
    text_qv_modules = [f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(36)] + \
                     [f"model.language_model.layers.{i}.self_attn.v_proj" for i in range(36)]
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=text_qv_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Create datasets
    train_dataset = SFT_Dataset(
        jsonl_path="./gqa_cot_train_10K.jsonl",
        image_dir="./images",
        processor=processor
    )
    eval_dataset = SFT_Dataset(
        jsonl_path="./gqa_cot_val_1K.jsonl",
        image_dir="./images",
        processor=processor
    )

    # Custom collate function
    def custom_collate(features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [torch.tensor([1,16,16]) for f in features]

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            "pixel_values": torch.stack(pixel_values),
            "image_grid_thw": torch.stack(image_grid_thw)
        }
        return batch

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwen_vl_sft",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=1,
        bf16=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        report_to="wandb",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate,
        tokenizer=processor.tokenizer,
    )

    # Train the model
    trainer.train()
    
    # Save the model and processor
    model.save_pretrained("./qwen_vl_sft")
    processor.save_pretrained("./qwen_vl_sft")
    
    wandb.finish()

if __name__ == "__main__":
    train_sft()