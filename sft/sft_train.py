from trl import SFTConfig

# Import load_model from model.py
from model import load_model

from sft.sft_dataset import train_dataset, eval_dataset
import wandb
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info


def collate_fn(processor, examples):
    """
    Data collator to prepare a batch of examples.

    This function applies the chat template to texts, processes the images,
    tokenizes the inputs, and creates labels with proper masking.
    """
    # Apply chat template to each example (no tokenization here)
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    # Process visual inputs for each example
    image_inputs = [process_vision_info(example)[0] for example in examples]

    # Tokenize texts and images into tensors with padding
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # Create labels by cloning input_ids and mask the pad tokens
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Determine image token IDs to mask in the labels (model specific)
    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels

    return batch



def main():
    # Load model and processor using the refactored function
    model, processor = load_model(None)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="./qwen-sft-052814",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        learning_rate=2e-4,  # Learning rate for training
        logging_steps=1,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        load_best_model_at_end=True,  # Load the best model after training
        bf16=True,  # Use bfloat16 precision
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        report_to="wandb",  # Reporting tool for tracking metrics
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns = False
    )

    wandb.init(
        project="qwen2_5-vl-sft",  # change this
        name="qwen2_5-vl-sft",  # change this
        config=training_args,
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn(processor),
        peft_config=None,  # peft_config is now handled inside load_model
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # trainer.save_model(training_args.output_dir)
    # trainer.save_processor(training_args.output_dir)
    model.save_pretrained("./qwen_vl_sft_052814")
    processor.save_pretrained("./qwen_vl_sft_052814")
    trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    main()
