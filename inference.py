import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel
from PIL import Image

model_id = "google/gemma-3-4b-it"
adapter_dir = "./gemma_mm_sft"
image_path = "test.jpg"
question = "Who is wearing a shirt?"

processor = AutoProcessor.from_pretrained(adapter_dir)

base_model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True
)

model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

image = Image.open(image_path).convert("RGB")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Question: {question}"}
    ]}
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

input_len = inputs["input_ids"].shape[-1]
generation = generated_ids[0][input_len:]
output_text = processor.decode(generation, skip_special_tokens=True)

print("\n🧾 === 모델 출력 결과 ===")
print(output_text)