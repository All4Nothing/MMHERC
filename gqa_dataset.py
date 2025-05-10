import json
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor

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
        entry = self.data[idx]

        image_path = f"{self.image_dir}/{entry['image']}"
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {entry['question']}"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": entry['rationale']}]
            }
        ]

        input_data = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_data["input_ids"][0],
            "attention_mask": input_data["attention_mask"][0],
            "pixel_values": input_data["pixel_values"][0],
            "labels": input_data["input_ids"][0]
        }