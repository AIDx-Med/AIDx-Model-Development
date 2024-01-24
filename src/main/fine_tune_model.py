import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import os
from evaluate import load
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from src.processing.utils import transform_dataset_to_tensor, BatchPaddedCollator

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def main(args):
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_r = 8
    lora_alpha = 2 * lora_r
    lora_dropout = 0.1

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["w1", "w2", "w3"],  # just targeting the MoE layers.
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    parquet_dir = args.parquet_dir
    train_parquet_file = os.path.join(parquet_dir, "train.parquet")
    test_parquet_file = os.path.join(parquet_dir, "test.parquet")

    initial_dataset = load_dataset(
        "parquet", data_files=train_parquet_file, streaming=True, split="train"
    )
    dataset = initial_dataset.map(transform_dataset_to_tensor, batched=True)

    test_dataset = load_dataset(
        "parquet", data_files=test_parquet_file, streaming=True, split="test"
    )
    test_dataset = test_dataset.map(transform_dataset_to_tensor, batched=True)

    train_data = dataset["train"]
    test_data = test_dataset["test"]

    data_collator = BatchPaddedCollator(tokenizer, mlm=False)

    metric = load("bertscore")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, model_type='nfliu/scibert_basevocab_uncased')

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=6,
            learning_rate=2e-4,
            logging_steps=1,
            fp16=True,
            optim="paged_adamw_8bit",
            output_dir="aidx-mixtral",
            load_best_model_at_end=True,
        ),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    train_result = trainer.train()

    # save the best model as safetensors
    trainer.save_model("aidx-mixtral")

    metrics = train_result.metrics
    trainer.log_metrics("all", metrics)
    trainer.save_metrics("all", metrics)

    # evaluate the model using bert-score
    eval_result = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
    trainer.log_metrics("test", eval_result)
    trainer.save_metrics("test", eval_result)

    print_summary(train_result)

