from dotenv import load_dotenv
import os
from tqdm.auto import tqdm
import ray

from src.database.engine import create_sqlalchemy_engine, get_log_model
from src.database.logging import read_processed_hadm_ids, fetch_all_hadm_ids
from src.processing.workflow import (
    process_hadm_id_ray,
    process_discharge_note_ray,
)


load_dotenv("config/.env")
HOST_IP = os.environ["DATABASE_IP"]
DATABASE_USER = os.environ["DATABASE_USER"]
DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
DATABASE_PORT = os.environ["DATABASE_PORT"]


def main(args):
    rewrite_log_file = args.rewrite_log_db
    discharge_note_only = args.discharge_note_only

    engine = create_sqlalchemy_engine("mimiciv")
    mimicllm_engine = create_sqlalchemy_engine("mimicllm")

    log_model = get_log_model(
        log_table="logs" if not discharge_note_only else "discharge_note_logs"
    )

    processed_hadm_ids = read_processed_hadm_ids(
        mimicllm_engine, log_model, rewrite=rewrite_log_file
    )
    all_hadm_ids = fetch_all_hadm_ids(engine)
    hadm_ids = [
        hadm_id for hadm_id in all_hadm_ids if hadm_id not in processed_hadm_ids
    ]

    # dispose of the engine to avoid memory leaks
    engine.dispose()
    mimicllm_engine.dispose()

    context = ray.init(dashboard_host="0.0.0.0")
    print(f"Connected to Ray Dashboard at {context.dashboard_url}")

    # Set up the progress bar
    with tqdm(total=len(hadm_ids), desc="Processing", dynamic_ncols=True) as pbar:
        if discharge_note_only:
            futures = [
                process_discharge_note_ray.remote(hadm_id) for hadm_id in hadm_ids
            ]
        else:
            futures = [process_hadm_id_ray.remote(hadm_id) for hadm_id in hadm_ids]

        remaining_futures = set(futures)
        while remaining_futures:
            done_futures, remaining_futures = ray.wait(list(remaining_futures))
            for future in done_futures:
                try:
                    hadm_id, error = ray.get(future)
                    if error:
                        pbar.write(error)  # Ray takes care of orderly printing
                except Exception as e:
                    pbar.write(
                        f"Error processing hadm_id {hadm_id}: {type(e).__name__} errored with message: {e}"
                    )
                pbar.set_description(f"Completed hadm_id {hadm_id}")
                pbar.update(1)
