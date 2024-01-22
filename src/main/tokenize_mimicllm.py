import ray
from ray.experimental.tqdm_ray import tqdm
from transformers.utils.logging import disable_progress_bar

from src.processing.tokenization import batch_strings, get_all_sample_ids
from src.database.engine import create_sqlalchemy_engine, get_log_model
from src.database.logging import read_processed_hadm_ids
from src.processing.workflow import tokenize_batch_ray

disable_progress_bar()

remote_tqdm = ray.remote(tqdm)


def main(args):
    batch_size = args.batch_size
    rewrite_log_db = args.rewrite_log_db

    engine = create_sqlalchemy_engine("mimicllm")
    log_model = get_log_model("tokenization_logs")

    print("Fetching sample IDs...")
    all_sample_ids = get_all_sample_ids(engine)
    processed_sample_ids = read_processed_hadm_ids(
        engine, log_model, rewrite=rewrite_log_db
    )
    sample_ids = [
        sample_id
        for sample_id in all_sample_ids
        if sample_id not in processed_sample_ids
    ]
    print(f"Found {'{:,}'.format(len(sample_ids))} sample IDs to process")

    # dispose of the engine to avoid memory leaks
    engine.dispose()

    context = ray.init(dashboard_host="0.0.0.0")
    print(f"Connected to Ray Dashboard at {context.dashboard_url}")

    num_cpus = ray.available_resources()["CPU"]
    print(f"Number of CPUs available to Ray: {num_cpus}")

    print(f"Batch size: {batch_size}")

    # organize sample_ids into batches
    batched_sample_ids = batch_strings(sample_ids, batch_size // num_cpus)

    progress_actor = remote_tqdm.remote(total=len(sample_ids), desc="Tokenizing")

    # Set up the progress bar
    futures = [
        tokenize_batch_ray.remote(batch_ids, progress_actor)
        for batch_ids in batched_sample_ids
    ]

    remaining_futures = set(futures)
    while remaining_futures:
        done_futures, remaining_futures = ray.wait(list(remaining_futures))
        for future in done_futures:
            try:
                ray.get(future)
            except Exception as e:
                progress_actor.write.remote(
                    f"Error processing: {type(e).__name__} error with message: {e}"
                )
