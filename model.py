from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import torch
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from transformers import prepare_model_for_kbit_training


def load_model(pretrained_lora_path=None):
    """
    Load the model and processor. If pretrained_lora_path is provided, load the LoRA-continued model.
    Otherwise, load the base model for training.
    Returns (model, processor)
    """
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id)

    if pretrained_lora_path is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = prepare_model_for_kbit_training(model)
        for name, param in model.named_parameters():
            if name.startswith("visual"):
                param.requires_grad = False
        model = get_peft_model(model, lora_config)
        print(model.print_trainable_parameters())
        return model, processor
    else:
        from peft import PeftModel, PeftConfig
        peft_config = PeftConfig.from_pretrained(pretrained_lora_path)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model = prepare_model_for_kbit_training(base_model)
        model = PeftModel.from_pretrained(base_model, pretrained_lora_path, is_trainable=True)
        print(model.print_trainable_parameters())
        return model, processor