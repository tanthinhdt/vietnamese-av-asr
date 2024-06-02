import os
import torch
import argparse
from logging import Logger
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from utils import (
    get_default_arg_parser,
    get_logger,
    load_config,
    check_compatibility_with_bf16,
)


def get_args() -> argparse.Namespace:
    parser = get_default_arg_parser(
        description="Train a model with supervised fine-tuning",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    return parser.parse_args()


def collate_fn(batch):
    pass


def main(config: dict, logger: Logger) -> None:
    dataset = load_dataset(
        config["dataset_name"],
        cache_dir=os.path.join(os.getcwd(), "data", "external"),
    )
    logger.info(f"Loaded {dataset} dataset with {len(dataset)} samples")

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["use_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["use_nested_quant"],
    )
    if check_compatibility_with_bf16:
        logger.INFO("Your GPU supports bfloat16: accelerate training with bf16=True")
    else:
        logger.INFO(
            "Your GPU does not support bfloat16: accelerate training with bf16=False"
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map=config["device_map"],
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    logger.info(f'Loaded {config["model_name"]} model')

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        r=config["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    num_trainable_params, num_params = model.get_nb_trainable_parameters()
    logger.info(f"Model has {num_params} parameters")
    logger.info(f"Model has {num_trainable_params} trainable parameters")

    # Set training parameters
    training_arguments = TrainingArguments(
        do_train=config["do_train"],
        do_eval=config["do_eval"],
        evaluation_strategy="steps",
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim=config["optim"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        max_grad_norm=config["max_grad_norm"],
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        group_by_length=config["group_by_length"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to="tensorboard",
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config["packing"],
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    config = load_config()
    logger = get_logger(name="TrainModel", log_path=args.log_path)
    main(config=config, logger=logger)
