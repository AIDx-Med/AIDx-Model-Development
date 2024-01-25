from src.training.training_utils import (
    print_summary,
    eval_and_save_metrics,
    load_model_trainer,
    load_bertscore,
    load_data,
    create_data_collator,
    setup_training_env,
)


def main(args):
    base_model_id = args.model_name
    parquet_dir = args.parquet_dir

    bnb_config = setup_training_env()
    data_collator = create_data_collator(base_model_id)
    test_data, train_data, val_data = load_data(parquet_dir)
    compute_bertscore = load_bertscore()

    trainer = load_model_trainer(
        base_model_id,
        bnb_config,
        compute_bertscore,
        data_collator,
        train_data,
        val_data,
    )

    train_result = trainer.train()

    # save the best model adapter as safetensors
    trainer.save_model("aidx-mixtral")

    eval_and_save_metrics(test_data, train_result, trainer)

    print_summary(train_result)
