import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from gqa_dataset import GQADataset
from preprocess import prepare_dataset

model_id = "google/gemma-3-4b-it"
train_json_path = "data/gqa_cot_train.jsonl"
val_json_path = "data/gqa_cot_val.jsonl"
image_dir = "data/images"

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit = True,
    )

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, 
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

train_dataset = GQADataset("data/gqa_cot_train.jsonl", "data/images", processor)
val_dataset = GQADataset("data/gqa_cot_val.jsonl", "data/images", processor)

training_args = TrainingArguments(
    output_dir="./gemma_mm_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    dataloader_num_workers=8,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer
)

trainer.train()
model.save_pretrained("./gemma_mm_sft")
processor.save_pretrained("./gemma_mm_sft")