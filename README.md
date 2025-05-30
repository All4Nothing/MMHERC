# MMHERC (Mitigating Multimodal Hallucinations by Enhancing the Reasoning Capabilities)

This project implements a two-stage training approach to improve the reasoning capabilities of vision-language models and reduce hallucinations:
1. Supervised Fine-Tuning (SFT)
2. Group Relative Policy Optimization (GRPO)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/All4Nothing/MMHERC.git
cd MMHERC
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

Note: For Flash Attention support, install flash-attn separately:
```bash
pip install flash-attn --no-build-isolation
```

## Data Preparation

1. Prepare your dataset in the following format:
   - Training data: `gqa_cot_train_10k.jsonl`
   - Validation data: `gqa_cot_val_1k.jsonl`
   - Images should be placed in an `images` directory

2. The dataset should be in JSONL format with the following structure:
```json
{
    "image": "image_filename.jpg",
    "question": "What is the color of the car?",
    "thought": "I can see a red car in the image",
    "answer": "red"
}
```

## Training

### 1. Supervised Fine-Tuning (SFT)

To run SFT training:

```bash
python sft/sft_train.py
```

The training script will:
- Load the base Qwen2.5-VL model
- Apply LoRA fine-tuning
- Save checkpoints in the specified output directory

### 2. Group Relative Policy Optimization (GRPO)

To run GRPO training:

```bash
python grpo/grpo_train.py
```

The GRPO training:
- Uses the SFT-fine-tuned model as the base
- Implements two reward functions:
  - Semantic similarity reward
  - Format compliance reward
- Saves checkpoints in the specified output directory

## Configuration

### SFT Training Parameters
- Model: Qwen2.5-VL-3B-Instruct
- Batch size: 4
- Learning rate: 2e-4
- Training epochs: 3
- LoRA configuration:
  - r: 8
  - alpha: 16
  - target modules: ["q_proj", "v_proj"]

### GRPO Training Parameters
- Batch size: 16
- Learning rate: 1e-6
- Beta: 0.04
- Number of generations: 4
- Max prompt length: 1024
- Max completion length: 128

## Output Format

The model is trained to follow this exact format:
```
<thought>your reasoning</thought>
<answer>your answer(one or two words)</answer>
```

## Requirements

See `requirements.txt` for the complete list of dependencies. Key requirements include:
- torch >= 2.0.0
- transformers >= 4.39.0
- accelerate >= 0.27.0
- peft >= 0.7.0
- trl >= 0.7.0
- flash-attn >= 2.3.0

## License

[Add your license information here]
