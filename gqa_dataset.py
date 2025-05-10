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

        processed = self.processor.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )

        return {
            "input_ids": processed["input_ids"][0],
            "attention_mask": processed["attention_mask"][0],
            "pixel_values": processed["pixel_values"][0],
            "labels": processed["input_ids"][0],
        }