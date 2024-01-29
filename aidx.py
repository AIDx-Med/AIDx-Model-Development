import argparse
import sys
from dotenv import load_dotenv
import os

from src.main.create_mimicllm import main as create_mimicllm_main
from src.main.tokenize_mimicllm import main as tokenize_mimicllm_main
from src.main.create_parquet_datasets import main as create_parquet_datasets_main
from src.main.fine_tune_model import main as fine_tune_model_main


load_dotenv("config/.env")
HOST_IP = os.environ["DATABASE_IP"]
DATABASE_USER = os.environ["DATABASE_USER"]
DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
DATABASE_PORT = os.environ["DATABASE_PORT"]


def main():
    parser = argparse.ArgumentParser(description="AIDx Entry Point")
    subparsers = parser.add_subparsers(title="subcommands", required=True)

    # Define a subparser for each command
    create_mimicllm_parser = subparsers.add_parser(
        "create-mimicllm", help="Create the MIMIC-LLM database"
    )
    create_mimicllm_parser.add_argument(
        "--rewrite-log-db",
        action="store_true",
        help="If set, will rewrite the log database",
    )
    create_mimicllm_parser.add_argument(
        "--discharge-note-only",
        action="store_true",
        help="If set, will only process discharge notes",
    )
    create_mimicllm_parser.set_defaults(func=create_mimicllm_main)

    tokenize_mimicllm_parser = subparsers.add_parser(
        "tokenize-mimicllm", help="Tokenize the MIMIC-LLM database"
    )
    tokenize_mimicllm_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="The batch size to use for tokenization",
    )
    tokenize_mimicllm_parser.add_argument(
        "--rewrite-log-db",
        action="store_true",
        help="If set, will rewrite the log database",
    )
    tokenize_mimicllm_parser.set_defaults(func=tokenize_mimicllm_main)

    create_parquet_datasets_parser = subparsers.add_parser(
        "create-parquet-datasets", help="Create the Parquet train/test datasets"
    )
    create_parquet_datasets_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="The chunk size to use for querying the database",
    )
    create_parquet_datasets_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="The test size to use for splitting the data",
    )
    create_parquet_datasets_parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/parquet",
        help="The directory to store the Parquet files",
    )
    create_parquet_datasets_parser.set_defaults(func=create_parquet_datasets_main)

    fine_tune_model_parser = subparsers.add_parser(
        "fine-tune-model", help="Fine tune the model"
    )
    fine_tune_model_parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="The model name to use for fine tuning",
    )
    fine_tune_model_parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/mimicllm",
        help="The directory with your parquet files",
    )
    fine_tune_model_parser.add_argument(
        '--stream-data',
        action='store_true',
        help='If set, will stream the data from the parquet files',
    )
    fine_tune_model_parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="If set, will clear the cache",
    )
    fine_tune_model_parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="The maximum batch size to use for fine tuning",
    )
    fine_tune_model_parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="The number of epochs to use for fine tuning",
    )
    fine_tune_model_parser.set_defaults(func=fine_tune_model_main)

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    args.func(args)


if __name__ == "__main__":
    main()
