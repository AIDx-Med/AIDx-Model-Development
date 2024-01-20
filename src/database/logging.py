from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text


def fetch_all_hadm_ids(engine):
    query = text(
        """
    SELECT a.hadm_id
    FROM mimiciv_hosp.admissions a
    INNER JOIN mimiciv_ed.edstays e ON a.hadm_id = e.hadm_id
    WHERE EXTRACT(day FROM (a.dischtime - a.admittime)) < 7;
    """
    )

    with engine.connect() as connection:
        result = connection.execute(query).fetchall()

    list_results = [item[0] for item in result]

    return list_results


def read_processed_hadm_ids(mimicllm_engine, log_model, rewrite=False):
    session_maker = sessionmaker(bind=mimicllm_engine)
    session = session_maker()

    if rewrite:
        # Use TRUNCATE to efficiently clear the table
        session.execute(text("TRUNCATE TABLE mimicllm.logs;"))
        session.commit()

    try:
        # Query the database for all hadm_id values
        processed_hadm_ids = {
            log.hadm_id for log in session.query(log_model.hadm_id).all()
        }
        return processed_hadm_ids
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def log_hadm_id(hadm_id, mimicllm_engine, log_model):
    session_maker = sessionmaker(bind=mimicllm_engine)
    session = session_maker()

    new_log = log_model(hadm_id=hadm_id)
    session.add(new_log)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
    except Exception as e:
        session.rollback()

        raise e
    finally:
        session.close()
