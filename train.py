import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from preprocess import prepare_dataset
from torch.utils.data import Dataset

class VisualCoTDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}]
        messages += item["messages"]
        example = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            )
        return {
            "input_ids": example["input_ids"][0],
            "attention_mask": example["attention_mask"][0],
            "pixel_values": example["pixel_values"][0],
            "labels": example["labels"][0],
        }
    
if __name__ == "__main__":
    model_id = "google/gemma-3-4b-it"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit = True,
        )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = prepare_dataset()
    train_dataset = VisualCoTDataset(dataset, processor)

    training_args = TrainingArguments(
        output_dir="./gemma_mm_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer
    )

    trainer.train()
    model.save_pretrained("./gemma_mm_sft")
    processor.save_pretrained("./gemma_mm_sft")