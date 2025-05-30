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
    base_dir = "./images"
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

# Create and format the datasets
train_dataset = SFT_Dataset("./gqa_cot_train_10k.jsonl", "./images")
eval_dataset = SFT_Dataset("./gqa_cot_val_1k.jsonl", "./images")
train_dataset = [format_data(sample) for sample in tqdm(train_dataset, desc="Formatting train set", file=sys.stdout)]
eval_dataset = [format_data(sample) for sample in tqdm(eval_dataset, desc="Formatting eval set", file=sys.stdout)]

# Export the datasets for use in other scripts
__all__ = ['train_dataset', 'eval_dataset']
