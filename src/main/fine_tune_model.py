import os
import torch
import transformers
import sys

from transformers import AutoTokenizer

current_file_directory = os.path.dirname(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file_directory))
absolute_src_path = os.path.abspath(src_dir)

sys.path.append(absolute_src_path)

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


def main(args=None):
    if not args:
        base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        parquet_dir = "data/parquet"
        stream_data = False
        clear_cache = False
        max_batch_size = 1
        num_epochs = 2
    elif args:
        base_model_id = args.model_name or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        parquet_dir = args.parquet_dir or "data/parquet"
        stream_data = args.stream_data or False
        clear_cache = args.clear_cache or False
        max_batch_size = args.max_batch_size or 1
        num_epochs = args.num_epochs or 2

    # set up logging
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()

    print("Setting up training environment...")
    torch.set_num_threads(cpu_count//gpu_count)
    accelerator = setup_training_env()
    data_collator = create_data_collator(base_model_id)
    accelerator.print("Done setting up training environment...")

    if accelerator.is_main_process:
        transformers.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()

    accelerator.print(f"Number of CPUs available: {cpu_count}")
    accelerator.print(f"Number of GPUs available: {gpu_count}")

    accelerator.print("Loading data...")
    train_data, val_data, train_count, val_count = load_data(parquet_dir, stream=stream_data, cpu_count=cpu_count, accelerator=accelerator, clear_cache=clear_cache)
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
        max_batch_size,
        num_epochs,
        cpu_count//gpu_count,
        accelerator
    )

    print_gpu_utilization(accelerator)

    accelerator.print("Training model...")
    trainer.train()

    # save the best model adapter as safetensors
    trainer.save_model("aidx-mixtral")
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # restore padding side
    tokenizer.init_kwargs["padding_side"] = "left"
    tokenizer.save_pretrained('aidx-mixtral')

    accelerator.print("Done training model...")

if __name__ == "__main__":
    main()
