import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from trl import DPOTrainer, DPOConfig

set_seed(42)

model_id = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    attn_implementation="sdpa"
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.config.pad_token_id = tokenizer.pad_token_id

def filter_ambiguous_data(row):
    return row["is_response_0_safe"] != row["is_response_1_safe"]

def format_safety_data(row):
    prompt = row["prompt"]

    if row["is_response_0_safe"]:
        chosen = row["response_0"]
        rejected = row["response_1"]
    else:
        chosen = row["response_1"]
        rejected = row["response_0"]

    prompt_formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    chosen_full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
        tokenize=False
    )

    rejected_full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
        tokenize=False
    )

    return {
        "prompt": prompt_formatted,
        "chosen": chosen_full[len(prompt_formatted):],
        "rejected": rejected_full[len(prompt_formatted):],
    }

print("Incarcare dataset PKU-SafeRLHF...")
raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train[:3000]")

print("Se filtreaza datele ambigue...")
dataset = raw_dataset.filter(filter_ambiguous_data)

print("Se formateaza template-ul de chat...")
original_columns = dataset.column_names
dataset = dataset.map(format_safety_data, remove_columns=original_columns)

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)

dpo_config = DPOConfig(
    beta=0.4,
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    output_dir="./dpo_safety_results",
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    remove_unused_columns=False,
    run_name="dpo_safety_run",
    report_to="none",
    gradient_checkpointing=True
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("Start antrenare DPO...")
dpo_trainer.train()

print("Salvare model si tokenizer...")
dpo_trainer.save_model("./final_safe_llama")
tokenizer.save_pretrained("./final_safe_llama")