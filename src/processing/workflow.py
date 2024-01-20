import ray
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import traceback

from src.database.engine import create_sqlalchemy_engine, get_log_model
from src.database.logging import log_hadm_id
from src.database.queries import upload_to_db
from src.processing.data_transformation import patient_info_to_sample


def process_hadm_id(hadm_id, debug=False):
    engine = create_sqlalchemy_engine("mimiciv")
    mimicllm_engine = create_sqlalchemy_engine("mimicllm")

    log_model = get_log_model()

    query = text(
        """
    SELECT json_build_object(
        schema_name, json_agg(
            json_build_object(
                table_name, column_names
            )
        )
    )
    FROM (
        SELECT 
            t.table_schema as schema_name, 
            t.table_name, 
            json_agg(c.column_name ORDER BY c.ordinal_position) as column_names
        FROM information_schema.tables t
        INNER JOIN information_schema.columns c 
            ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog')
        GROUP BY t.table_schema, t.table_name
    ) AS sub
    GROUP BY schema_name;
    """
    )

    # Execute the query
    with engine.connect() as con:
        result = con.execute(query).fetchall()

    # Extract schemas and their respective tables with columns from the query result
    schemas_with_tables = [schema_result[0] for schema_result in result]

    # Flatten the list of dictionaries for each schema
    database_structure = {}
    for schema in schemas_with_tables:
        for schema_name, tables in schema.items():
            # Initialize the schema in the flattened dictionary if not already present
            if schema_name not in database_structure:
                database_structure[schema_name] = {}

            # Combine the tables under the same schema
            for table in tables:
                database_structure[schema_name].update(table)

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
