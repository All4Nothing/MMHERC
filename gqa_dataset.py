import json
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
import os

class GQADataset(Dataset):
    def __init__(self, jsonl_path, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.data = []

        with open(jsonl_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.data.append(ex)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert("RGB")

        output_text = (
            f"Thought: {item['thought'].strip()}\n\n"
            f"Full answer: {item['full_answer'].strip()}\n\n"
            f"Short answer: {item['answer'].strip()}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {item['question'].strip()}"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output_text}]
            }
        ]

        # === Processor는 string prompt만 반환 ===
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # === Tokenize prompt → input_ids, attention_mask ===
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0)
        }