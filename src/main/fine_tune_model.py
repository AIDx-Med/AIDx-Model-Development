import os
import torch
import transformers

from src.training.training_utils import (
    print_summary,
    eval_and_save_metrics,
    load_model_trainer,
    load_bertscore,
    load_data,
    create_data_collator,
    setup_training_env,
    print_gpu_utilization
)


def main(args):
    transformers.logging.set_verbosity_debug()

    base_model_id = args.model_name
    parquet_dir = args.parquet_dir
    stream_data = args.stream_data

    # set up logging
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()

    print("Setting up training environment...")
    accelerator = setup_training_env()
    data_collator = create_data_collator(base_model_id)
    accelerator.print("Done setting up training environment...")

    accelerator.print(f"Number of CPUs available: {cpu_count}")
    accelerator.print(f"Number of GPUs available: {gpu_count}")

    accelerator.print("Loading data...")
    train_data, val_data, train_count, val_count = load_data(parquet_dir, stream=stream_data, cpu_count=cpu_count, accelerator=accelerator)
    accelerator.print("Done loading data...")

    accelerator.print("Loading compute_bertscore...")
    compute_bertscore = load_bertscore()
    accelerator.print("Done loading compute_bertscore...")

    accelerator.print("Loading model trainer...")
    trainer = load_model_trainer(
        base_model_id,
        compute_bertscore,
        data_collator,
        train_data,
        val_data,
        train_count,
        cpu_count//gpu_count,
        accelerator
    )

    print_gpu_utilization(accelerator)

    accelerator.print("Training model...")
    train_result = trainer.train()

    # save the best model adapter as safetensors
    trainer.save_model("aidx-mixtral")

    accelerator.print("Done training model...")

    accelerator.print("Evaluating model...")
    test_data = load_data(parquet_dir, stream=True, cpu_count=cpu_count//gpu_count, test_only=True, accelerator=accelerator)

    eval_and_save_metrics(test_data, train_result, trainer)

    print_summary(train_result, accelerator)
