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
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{example['question']}"}, # \n\nYou must follow this EXACT format:\n<thought>your reasoning</thought>\n<answer>one word</answer>\n\nRules:\n1. You must use both <thought> and <answer> tags\n2. Your answer must be exactly ONE WORD\n3. Do not add any text outside these tags\n4. Do not use any other tags or formats"
                    ],
                },
            ),
        }
