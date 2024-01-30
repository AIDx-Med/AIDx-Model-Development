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
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
)

from src.processing.utils import transform_dataset_from_pickle

def print_trainable_parameters(model, accelerator):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def print_gpu_utilization(accelerator):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    accelerator.print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result, accelerator):
    accelerator.print(f"Time: {result.metrics['train_runtime']:.2f}")
    accelerator.print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization(accelerator)


def eval_and_save_metrics(test_data, train_result, trainer):
    metrics = train_result.metrics
    trainer.log_metrics("all", metrics)
    trainer.save_metrics("all", metrics)
    # evaluate the model using bert-score
    eval_result = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
    trainer.log_metrics("test", eval_result)
    trainer.save_metrics("test", eval_result)


def load_model_trainer(
    base_model_id, compute_bertscore, data_collator, train_data, val_data, max_batch_size, num_epochs, cpu_count, accelerator
):
    accelerator.print("Loading model...")

    project = "aidx-finetune"
    base_model_name = "mixtral"
    run_name = (
            base_model_name
            + "-"
            + project
            + "-"
            + datetime.now().strftime("%Y-%m-%d-%H-%M")
    )
    output_dir = "./training/" + run_name

    batch_size = max_batch_size
    gradient_accumulation_steps = 2

    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        dataloader_num_workers=cpu_count,
        learning_rate=0.0002,
        # lr_scheduler_type='cosine',
        bf16=True,
        # optim="adamw_bnb_8bit",
        logging_steps=1,
        logging_dir="./logs",  # Directory for storing logs
        save_strategy="steps",
        save_steps=5,  # Save checkpoints every 5 steps
        evaluation_strategy="epoch",
        do_eval=True,
        report_to="wandb",  # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        gradient_checkpointing_kwargs={
            'use_reentrant': False
        },
        deepspeed="/workspace/zero2.json"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True,
        )

    model.gradient_checkpointing_enable()
    accelerator.wait_for_everyone()

    accelerator.print("Done loading model...")

    accelerator.print("Preparing model for kbit training...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=[
            "gate",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "w1",
            "w2",
            "w3",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    accelerator.print("Loading LORA...")
    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model, accelerator)


    # Settings from axolotl
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
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


def load_data(parquet_dir, stream=True, val_size=0.1, cpu_count=1, test_only=False, clear_cache=False, accelerator=None):
    train_parquet_file = os.path.join(parquet_dir, "train.parquet")
    test_parquet_file = os.path.join(parquet_dir, "test.parquet")

    if stream:
        cpu_count = 1 # multiprocessing is not supported with streaming datasets

    if not test_only:
        with accelerator.main_process_first():
            init_train_dataset = load_dataset(
                "parquet", data_files=train_parquet_file, streaming=stream, split="train"
            )

            train_dataset = init_train_dataset.map(transform_dataset_from_pickle, batched=True, batch_size=1_000,
                                                   num_proc=cpu_count)

        accelerator.wait_for_everyone()

        if clear_cache:
            train_dataset.cleanup_cache_files()
    else:
        with accelerator.main_process_first():
            init_test_dataset = load_dataset(
                "parquet", data_files=test_parquet_file, streaming=stream, split="train"
            )

            test_data = init_test_dataset.map(transform_dataset_from_pickle, batched=True, batch_size=1_000,
                                              num_proc=cpu_count)

        accelerator.wait_for_everyone()

        if clear_cache:
            test_data.cleanup_cache_files()

        return test_data

    if stream:
        accelerator.print("Getting dataset lengths...")
        len_train = 0

        if accelerator.is_main_process:
            with tqdm(desc="Count") as pbar:
                for _ in train_dataset:
                    pbar.update(1)
                    len_train += 1

            len_train_tensor = torch.tensor(len_train)
        else:
            len_train_tensor = torch.tensor(0)

        accelerator.broadcast(len_train_tensor, src=0)

        len_train = len_train_tensor.item()

        accelerator.print(f"Original data size: {len_train:,} rows")

        accelerator.print("Shuffling training data...")
        shuffled_train_dataset = train_dataset.shuffle(buffer_size=10_000)

        accelerator.print("Splitting training data...")

        val_count = int(len_train * val_size)
        train_count = len_train - val_count

        accelerator.print(f"Training data size: {train_count:,} rows")
        accelerator.print(f"Validation data size: {val_count:,} rows")

        # split training data into training and validation
        train_data = shuffled_train_dataset.skip(val_count)
        val_data = shuffled_train_dataset.take(val_count)
    else:
        training_dataset = train_dataset.train_test_split(test_size=val_size)
        train_data = training_dataset["train"]
        val_data = training_dataset["test"]
        train_count = len(train_data)
        val_count = len(val_data)

    return train_data, val_data, train_count, val_count


def create_data_collator(base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return data_collator


def setup_training_env():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        ),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    # accelerator = Accelerator()
    wandb.login()
    wandb_project = "aidx-finetune"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    return accelerator
