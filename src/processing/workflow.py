import pandas as pd
import ray
from sqlalchemy.exc import IntegrityError
import traceback
import pickle

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.database.engine import create_sqlalchemy_engine, get_log_model
from src.database.logging import log_hadm_id
from src.database.queries import upload_to_db, get_mimiciv_schema
from src.processing.data_transformation import patient_info_to_sample
from src.processing.timeline_generation import format_df_to_text
from src.processing.tokenization import (
    get_data,
    extract_base_id,
    extract_numeric_id,
    generate_prompt,
    tokenize,
)


def process_hadm_id(hadm_id, debug=False):
    engine = create_sqlalchemy_engine("mimiciv")
    mimicllm_engine = create_sqlalchemy_engine("mimicllm")

    log_model = get_log_model()

    database_structure = get_mimiciv_schema(engine)

    # main code
    try:
        df = patient_info_to_sample(hadm_id, database_structure, engine)
        upload_to_db(df, mimicllm_engine)
        log_hadm_id(hadm_id, mimicllm_engine, log_model)  # Log the processed hadm_id

        engine.dispose()
        mimicllm_engine.dispose()

        return hadm_id, False
    except IntegrityError:
        # If the hadm_id has already been processed, log the error and continue
        log_hadm_id(hadm_id, mimicllm_engine, log_model)

        engine.dispose()
        mimicllm_engine.dispose()

        return hadm_id, False
    except Exception as e:
        error_message = f"Error processing hadm_id {hadm_id}: {type(e).__name__} errored with message: {e}"
        if debug:
            error_message = (
                f"Error processing hadm_id {hadm_id}:\n{traceback.format_exc()}"
            )

        engine.dispose()
        mimicllm_engine.dispose()

        return hadm_id, error_message


@ray.remote
def process_hadm_id_ray(hadm_id, debug=False):
    return process_hadm_id(hadm_id, debug=debug)


def process_discharge_note(hadm_id, debug=False):
    engine = create_sqlalchemy_engine("mimiciv")
    mimicllm_engine = create_sqlalchemy_engine("mimicllm")

    log_model = get_log_model(log_table="discharge_note_logs")

    database_structure = get_mimiciv_schema(engine)

    # main code
    try:
        df, subject_id = patient_info_to_sample(
            hadm_id, database_structure, engine, discharge_note_only=True
        )
        df = df.rename(columns={"diagnoses": "Potential diagnoses"})

        input_text = format_df_to_text(df[["patient information", "discharge note"]])
        input_text += "\n\n What diagnoses should be considered?"

        output_text = format_df_to_text(df[["Potential diagnoses"]])

        discharge_df = pd.DataFrame(
            [
                {
                    "sample_id": f"{hadm_id}-discharge",
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "input": input_text,
                    "output": output_text,
                }
            ]
        )

        upload_to_db(discharge_df, mimicllm_engine)
        log_hadm_id(hadm_id, mimicllm_engine, log_model)  # Log the processed hadm_id

        engine.dispose()
        mimicllm_engine.dispose()

        return hadm_id, False
    except IntegrityError:
        # If the hadm_id has already been processed, log the error and continue
        log_hadm_id(hadm_id, mimicllm_engine, log_model)

        engine.dispose()
        mimicllm_engine.dispose()

        return hadm_id, False
    except Exception as e:
        error_message = f"Error processing hadm_id {hadm_id}: {type(e).__name__} errored with message: {e}"
        if debug:
            error_message = (
                f"Error processing hadm_id {hadm_id}:\n{traceback.format_exc()}"
            )

        engine.dispose()
        mimicllm_engine.dispose()

        return hadm_id, error_message


@ray.remote
def process_discharge_note_ray(hadm_id, debug=False):
    return process_discharge_note(hadm_id, debug=debug)


def tokenize_batch(batch_ids, progress_actor=None):
    max_length = 32_000

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    engine = create_sqlalchemy_engine("mimicllm")

    system_prompt = ""

    log_model = get_log_model("tokenization_logs")

    batch_data = get_data(batch_ids, engine)

    last_skipped_id = {}
    tokenized_prompts = []

    if progress_actor is None:
        pbar = tqdm(total=len(batch_data), desc="Tokenizing", dynamic_ncols=True)

    for index, row in batch_data.iterrows():
        log_hadm_id(row["sample_id"], engine, log_model)
        base_id = extract_base_id(row["sample_id"])
        numeric_id = extract_numeric_id(row["sample_id"])
        # Skip logic for non-discharge samples
        if base_id in last_skipped_id and numeric_id is not None:
            if numeric_id >= last_skipped_id[base_id] and not row["sample_id"].endswith(
                "discharge"
            ):
                continue

        prompt = generate_prompt(system_prompt, row["input"], row["output"])
        tokenized = tokenize(prompt, tokenizer)

        if progress_actor is None:
            pbar.set_description(f"Samples - {row['sample_id']}")
            pbar.set_postfix_str(f"Length: {len(tokenized['input_ids'][0])}")

        if len(tokenized["input_ids"][0]) > max_length:
            if numeric_id is not None:
                last_skipped_id[base_id] = numeric_id
        else:
            tokenized_prompts.append(tokenized)

        if progress_actor is None:
            pbar.update(1)
        else:
            progress_actor.update.remote(1)

    serialized_tokens = pd.DataFrame(tokenized_prompts).map(lambda x: pickle.dumps(x))
    upload_to_db(serialized_tokens, engine, table="tokenized_data")

    if progress_actor is None:
        pbar.close()
    engine.dispose()


@ray.remote
def tokenize_batch_ray(batch_ids, progress_actor):
    return tokenize_batch(batch_ids, progress_actor)
