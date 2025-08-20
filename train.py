import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from huggingface_hub import login

def main():
    # --- 1. LOGIN TO SERVICES ---
    # For local development, it's better to log in via the terminal:
    # `huggingface-cli login` and `wandb login`
    # However, we can use environment variables for automation.
    
    hf_token = os.environ.get("HF_TOKEN")
    wandb_token = os.environ.get("WANDB_API_KEY")

    if not hf_token or not wandb_token:
        print("Please set HF_TOKEN and WANDB_API_KEY environment variables.")
        return
        
    wandb.login(key=wandb_token)
    login(token=hf_token)
    print("Login successful.")

    # --- 2. DEFINE MODELS AND DATASET ---
    base_model_id = "google/gemma-2b-it"
    dataset_id = "plameneduardo/code-alpaca-20k"
    
    # --- 3. LOAD DATASET AND PREPARE ---
    print("Loading and preparing dataset...")
    dataset = load_dataset("json", data_files=f"Code_Alpaca_20k.json", split="train") 

    def format_and_tokenize(example):
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        result = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result
        
    # --- 4. LOAD MODEL AND TOKENIZER ---
    print("Loading model and tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={'': 0},
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    tokenized_dataset = dataset.map(format_and_tokenize)
    print("Dataset formatted and tokenized.")

    # --- 5. CONFIGURE AND RUN TRAINING ---
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="gemma-code-explainer-final",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        save_steps=200,
        report_to="wandb",
        run_name="gemma-2b-code-alpaca-finetune-final",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")
    
    # --- 6. SAVE AND PUSH FINAL MODEL ---
    final_model_name = "mohan1201/gemma-2b-code-explainer-final" 
    print(f"Saving and pushing final model to {final_model_name}...")
    trainer.push_to_hub(final_model_name)
    print("Process complete!")

if __name__ == "__main__":
    main()
