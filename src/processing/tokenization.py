import pickle

from sqlalchemy import text as sql_text
import pandas as pd


def tokenize(text, tokenizer):
    return tokenizer(text, return_tensors="pt")


def generate_prompt(system, input_text, output_text, separate=False):
    # convert to instruction formatting
    input_prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{input_text}\n<|im_end|>\n<|im_start|>assistant\n"
    output_prompt = f"{output_text}\n<|im_end|></s>"

    if separate:
        return {"input": input_prompt, "output": output_prompt}

    return input_prompt + output_prompt


def format_batch_query(batch, start_at=0):
    return sql_text(
        f"""
    SELECT sample_id, input, output
    FROM mimicllm.data
    ORDER BY sample_id
    LIMIT {batch}
    OFFSET {start_at}
    """
    )


def get_batch(batch_size, engine, start_at=0):
    query = format_batch_query(batch_size, start_at)
    df = pd.read_sql(query, engine)
    return df


def get_data(sample_ids, engine):
    query = sql_text(
        f"""
    SELECT sample_id, input, output
    FROM mimicllm.data
    WHERE sample_id IN ({', '.join([f"'{sample_id}'" for sample_id in sample_ids])})
    ORDER BY sample_id
    """
    )
    df = pd.read_sql(query, engine)
    return df


def extract_base_id(sample_id):
    """Extract the base ID from the sample ID."""
    if sample_id.endswith("discharge"):
        return sample_id.replace("_discharge", "")
    return "_".join(sample_id.split("_")[:-1])


def extract_numeric_id(sample_id):
    """Extract the numeric part of the sample ID."""
    parts = sample_id.split("_")
    if parts[-1].isdigit():
        return int(parts[-1])
    return None  # For 'discharge' or other non-numeric parts


def batch_strings(string_list, batch_size):
    # Initialize the list of batches and the current batch
    batches = []
    current_batch = []

    # Iterate over each string in the list
    for string in string_list:
        # Add string to the current batch
        current_batch.append(string)

        # If the current batch reaches the batch size, add it to the batches list
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []  # Start a new batch

    # Add the last batch if it contains any strings
    if current_batch:
        batches.append(current_batch)

    return batches

def get_all_sample_ids(engine):
    query = sql_text(
        """
    SELECT sample_id
    FROM mimicllm.data
    ORDER BY sample_id
    """
    )
    df = pd.read_sql(query, engine)
    return df["sample_id"].tolist()

def serialize_tokenized_row(row):
    attn_mask = pickle.dumps(row["attention_mask"])
    input_ids = pickle.dumps(row["input_ids"])
    return pd.Series({
        "token_id": row["sample_id"],
        "attention_mask": attn_mask,
        "input_ids": input_ids,
        "token_count": row["token_count"],
        "valid": row["valid"]
    })
