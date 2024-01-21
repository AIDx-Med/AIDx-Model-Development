import os
from sqlalchemy.orm import sessionmaker
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.database.engine import create_sqlalchemy_engine, create_tokenized_data_model


def main(args):
    chunk_size = args.chunk_size
    test_size = args.test_size
    parquet_dir = args.parquet_dir

    engine = create_sqlalchemy_engine("mimicllm")
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    tokenized_data_model = create_tokenized_data_model()
    total_rows = session.query(tokenized_data_model).count()

    os.makedirs(parquet_dir, exist_ok=True)
    train_parquet_file = os.path.join(parquet_dir, "train.parquet")
    test_parquet_file = os.path.join(parquet_dir, "test.parquet")

    last_id = 0

    # Initialize Parquet writers
    train_writer = None
    test_writer = None

    with tqdm(total=total_rows, desc="Processing") as pbar:
        while True:
            # Query the database in chunks ordered by a unique column
            query = (
                session.query(tokenized_data_model)
                .order_by(tokenized_data_model.token_id)
                .filter(tokenized_data_model.token_id > last_id)
                .limit(chunk_size)
            )
            chunk = pd.read_sql(query.statement, session.bind).drop(
                columns=["token_id"]
            )

            if chunk.empty:
                break

            train_chunk, test_chunk = train_test_split(chunk, test_size=test_size)

            # Convert DataFrame to PyArrow Table
            train_table = pa.Table.from_pandas(train_chunk, preserve_index=False)
            test_table = pa.Table.from_pandas(test_chunk, preserve_index=False)

            # Write train chunk
            if train_writer is None:
                train_writer = pq.ParquetWriter(
                    train_parquet_file, train_table.schema, compression="snappy"
                )
            train_writer.write_table(train_table)

            # Write test chunk
            if test_writer is None:
                test_writer = pq.ParquetWriter(
                    test_parquet_file, test_table.schema, compression="snappy"
                )

            test_writer.write_table(test_table)

            last_id = int(chunk["token_id"].iloc[-1])

            # Update progress bar
            pbar.update(len(chunk))

    # Close the Parquet writers
    if train_writer:
        train_writer.close()
    if test_writer:
        test_writer.close()

    session.close()
    engine.dispose()
