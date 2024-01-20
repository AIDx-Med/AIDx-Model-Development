from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os


# Define the ORM model within a function to avoid serialization issues
def get_log_model(log_table="logs"):
    from sqlalchemy import Column, BigInteger
    from sqlalchemy.orm import declarative_base

    base = declarative_base()

    class Log(base):
        __tablename__ = log_table
        __table_args__ = {"schema": "mimicllm"}
        hadm_id = Column(BigInteger, primary_key=True)

        def __init__(self, hadm_id):
            self.hadm_id = hadm_id

    return Log


# Define a function to create and return a new SQLAlchemy engine
def create_sqlalchemy_engine(database):
    connection_url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["DATABASE_USER"],
        password=os.environ["DATABASE_PASSWORD"],
        host=os.environ["DATABASE_IP"],
        port=int(os.environ["DATABASE_PORT"]),
        database=database,
    )
    return create_engine(connection_url)
