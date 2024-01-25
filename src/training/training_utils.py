import os
from datetime import datetime

import numpy as np
import torch
import wandb
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datasets import load_dataset
from evaluate import load
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    BitsAndBytesConfig,
)

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


def eval_and_save_metrics(test_data, train_result, trainer):
    metrics = train_result.metrics
    trainer.log_metrics("all", metrics)
    trainer.save_metrics("all", metrics)
    # evaluate the model using bert-score
    eval_result = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
    trainer.log_metrics("test", eval_result)
    trainer.save_metrics("test", eval_result)


def load_model_trainer(
    base_model_id, bnb_config, compute_bertscore, data_collator, train_data, val_data
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="cuda"
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "w1",
            "w2",
            "w3",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    project = "aidx-finetune"
    base_model_name = "mixtral"
    run_name = (
        base_model_name
        + "-"
        + project
        + "-"
        + datetime.now().strftime("%Y-%m-%d-%H-%M")
    )
    output_dir = "./" + run_name
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=8,
            auto_find_batch_size=True,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=3,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            fp16=True,
            optim="paged_adamw_8bit",
            logging_steps=1,  # When to start reporting loss
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=25,  # Save checkpoints every 25 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=25,  # Evaluate and save checkpoints every 25 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to="wandb",  # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        ),
        data_collator=data_collator,
        compute_metrics=compute_bertscore,
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    return trainer


def load_bertscore():
    metric = load("bertscore")

    def compute_bertscore(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(
            predictions=predictions,
            references=labels,
            model_type="nfliu/scibert_basevocab_uncased",
        )

    return compute_bertscore


def load_data(parquet_dir):
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
    # split training data into training and validation
    train_data, val_data = train_data.train_test_split(test_size=0.1)
    test_data = test_dataset["test"]
    return test_data, train_data, val_data


def create_data_collator(base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    data_collator = BatchPaddedCollator(tokenizer, mlm=False)
    return data_collator


def setup_training_env():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        ),
    )
    Accelerator(fsdp_plugin=fsdp_plugin)
    wandb.login()
    wandb_project = "aidx-finetune"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config
