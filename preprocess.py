from datasets import load_dataset
import os
import requests

def download_images(dataset, image_dir="data/images"):
    os.makedirs(image_dir, exist_ok=True)
    for example in dataset:
        url = example["image"]
        filename = os.path.join(image_dir, os.path.basename(url))
        if not os.path.exists(filename):
            img_data = requests.get(url).content
            with open(filename, "wb") as f:
                f.write(img_data)

def preprocess_for_mm(example):
    # Gemma3 멀티모달 입력 메시지 구성
    image_path = f"data/images/{os.path.basename(example['image'])}"
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Question: {example['question']}"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rationale"]}]
        }
    ]
    return example

def prepare_dataset():
    dataset = load_dataset("deepcs233/Visual-CoT", split="train")
    download_images(dataset)
    dataset = dataset.map(preprocess_for_mm)
    return dataset