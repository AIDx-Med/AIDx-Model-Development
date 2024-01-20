import argparse
import sys

from src.main.create_mimicllm import main as create_mimicllm_main

def main():
    parser = argparse.ArgumentParser(
        description="AIDx Entry Point"
    )
    subparsers = parser.add_subparsers(title="subcommands", required=True)

    # Define a subparser for each command
    create_mimicllm_parser = subparsers.add_parser("create-mimicllm", help="Create the MIMIC-LLM database")

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

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    args.func(args)

if __name__ == "__main__":
    main()