from sqlalchemy import text
import pandas as pd
from src.processing.utils import reorder_columns


def get_mimiciv_schema(engine):
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
    return database_structure


def query_hosp(hadm_id, database_structure, engine, diagnoses_only=False):
    query = text(
        "select * from mimiciv.mimiciv_hosp.admissions where hadm_id = :hadm;"
    ).bindparams(hadm=hadm_id)

    hospital_stay = pd.read_sql_query(query, engine)

    subject_id = int(hospital_stay["subject_id"][0])

    hosp_tables_to_ignore = [
        "admissions",
        "drgcodes",
        "emar",
        "hcpcsevents",
        "pharmacy",
    ]

    hosp_cols_to_remove = {
        "main": [
            "subject_id",
            "hadm_id",
            "admit_provider_id",
            "insurance",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ],
        "diagnoses_icd": ["subject_id", "hadm_id", "seq_num"],
        "labevents": [
            "subject_id",
            "hadm_id",
            "labevent_id",
            "order_provider_id",
            "storetime",
            "specimen_id",
        ],
        "microbiologyevents": [
            "subject_id",
            "hadm_id",
            "order_provider_id",
            "storetime",
            "microevent_id",
            "micro_specimen_id",
            "spec_itemid",
            "test_seq",
            "test_itemid",
            "chartdate",
            "storedate",
            "storetime",
            "org_itemid",
            "ab_itemid",
        ],
        "poe": [
            "subject_id",
            "hadm_id",
            "order_provider_id",
            "order_status",
            "poe_seq",
        ],
        "prescriptions": [
            "subject_id",
            "hadm_id",
            "pharmacy_id",
            "order_provider_id",
            "gsn",
            "ndc",
            "formulary_drug_cd",
            "poe_seq",
        ],
        "procedures_icd": ["subject_id", "hadm_id", "seq_num"],
        "services": ["subject_id", "hadm_id"],
        "transfers": ["subject_id", "hadm_id", "transfer_id"],
        "omr": ["subject_id", "seq_num"],
        "patients": ["subject_id", "anchor_year_group"],
    }

    for column, v in database_structure["mimiciv_hosp"].items():
        if "hadm_id" in v:
            if column in hosp_tables_to_ignore:
                continue

            if diagnoses_only and column != "diagnoses_icd":
                continue

            sql_query = text(
                f"select * from mimiciv.mimiciv_hosp.{column} where hadm_id = :hadm_id"
            ).bindparams(hadm_id=hadm_id)
            sql_df = pd.read_sql(sql_query, engine)

            removed_cols = hosp_cols_to_remove[column]
            sql_df = sql_df.drop(removed_cols, axis=1, errors="ignore")

            hospital_stay[column] = [sql_df]

    hospital_stay = hospital_stay.drop(
        hosp_cols_to_remove["main"], axis=1, errors="ignore"
    )

    # additional tables from subject id
    additional_patient_tables = ["omr", "patients"]

    for table in additional_patient_tables:
        sql_query = text(
            f"select * from mimiciv.mimiciv_hosp.{table} where subject_id = :subject_id"
        ).bindparams(subject_id=subject_id)
        sql_df = pd.read_sql(sql_query, engine)
        removed_cols = hosp_cols_to_remove[table]
        sql_df = sql_df.drop(removed_cols, axis=1, errors="ignore")

        hospital_stay[table] = [sql_df]

    return hospital_stay, subject_id


def query_ed(hadm_id, database_structure, engine):
    ed_stay_query = text(
        "select * from mimiciv.mimiciv_ed.edstays where hadm_id = :hadm_id"
    ).bindparams(hadm_id=hadm_id)

    # stay_id = 32522732
    # ed_stay_query = text("select * from mimiciv.mimiciv_ed.edstays where stay_id = :stay_id").bindparams(stay_id=stay_id)

    ed_stay_df = pd.read_sql(ed_stay_query, engine)

    ed_stays = []

    ed_cols_to_remove = {
        "main": ["subject_id", "race", "gender", "stay_id"],
        "diagnosis": ["subject_id", "stay_id", "seq_num", "icd_code", "icd_version"],
        "medrecon": ["subject_id", "stay_id", "gsn", "ndc", "etc_rn", "etccode", "gsn"],
        "pyxis": ["subject_id", "stay_id", "gsn", "med_rn", "gsn", "gsn_rn"],
        "triage": ["subject_id", "stay_id"],
        "vitalsign": ["subject_id", "stay_id"],
    }

    for row in ed_stay_df.iterrows():
        row_info = pd.DataFrame(row[1]).transpose()
        stay_id = row_info["stay_id"].values[0]

        for column, value in database_structure["mimiciv_ed"].items():
            if "stay_id" in value:
                if column == "edstays":
                    continue

                sql_query = text(
                    f"select * from mimiciv.mimiciv_ed.{column} where stay_id = :stay_id"
                ).bindparams(stay_id=stay_id)
                sql_df = pd.read_sql(sql_query, engine)

                removed_cols = ed_cols_to_remove[column]
                sql_df = sql_df.drop(removed_cols, axis=1, errors="ignore")

                match column:
                    case "triage":
                        # reorder and rename columns
                        sql_df = sql_df.rename(
                            columns={
                                "heartrate": "heart rate",
                                "resprate": "respiratory rate",
                                "o2sat": "oxygen saturation",
                                "sbp": "systolic blood pressure",
                                "dbp": "diastolic blood pressure",
                                "chiefcomplaint": "chief complaint",
                                "acuity": "emergency severity index",
                            }
                        )

                        sql_df = reorder_columns(
                            sql_df,
                            [
                                "chief complaint",
                                "pain",
                                "emergency severity index",
                                "temperature",
                                "heart rate",
                                "respiratory rate",
                                "oxygen saturation",
                                "systolic blood pressure",
                                "diastolic blood pressure",
                            ],
                        )
                    case "vitalsign":
                        # rename columns
                        sql_df = sql_df.rename(
                            columns={
                                "charttime": "chart time",
                                "heartrate": "heart rate",
                                "resprate": "respiratory rate",
                                "o2sat": "oxygen saturation",
                                "sbp": "systolic blood pressure",
                                "dbp": "diastolic blood pressure",
                            }
                        )
                    case "diagnosis":
                        # convert to a list of icd_title column
                        sql_df = sql_df["icd_title"].tolist()
                    case "medrecon":
                        sql_df = sql_df.drop("charttime", axis=1, errors="ignore")
                        sql_df = sql_df.rename(
                            columns={"etcdescription": "enhanced therapeutic class"}
                        )
                    case "pyxis":
                        # remove duplicate rows if `name` AND `charttime` are the same
                        sql_df = sql_df.drop_duplicates(subset=["name", "charttime"])
                        sql_df = sql_df.rename(
                            columns={
                                "charttime": "chart time",
                            }
                        )

                # if isinstance(sql_df, pd.DataFrame):
                #     sql_df = to_clean_records(sql_df)

                row_info[column] = [sql_df]
        ed_stays.append(row_info)

    ed_stays = pd.concat(ed_stays, ignore_index=True)

    ed_stays = ed_stays.drop(ed_cols_to_remove["main"], axis=1, errors="ignore")

    # rename columns to be more readable
    ed_stays = ed_stays.rename(
        columns={
            "intime": "arrival time",
            "outtime": "exit time",
            "arrival_transport": "arrival transport",
            "medrecon": "medication reconciliation",
            "pyxis": "dispensed medications",
            "vitalsign": "vital signs",
        }
    )

    # reorder columns
    ed_stays = reorder_columns(
        ed_stays,
        [
            "arrival time",
            "arrival transport",
            "triage",
            "vital signs",
            "medication reconciliation",
            "dispensed medications",
            "exit time",
            "disposition",
            "diagnosis",
        ],
    )

    return ed_stays


def query_discharge_note(engine, hadm_id, hospital_stay=None):
    discharge_note_query = text(
        "select * from mimiciv.mimiciv_note.discharge where hadm_id = :hadm_id"
    ).bindparams(hadm_id=hadm_id)

    discharge_note_df = pd.read_sql(discharge_note_query, engine)

    discharge_note_df = discharge_note_df.drop(
        ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime"],
        axis=1,
    )

    discharge_note_df = discharge_note_df.rename(
        columns={"storetime": "chart time", "text": "discharge note"}
    )

    if hospital_stay is not None:
        hospital_stay["discharge note"] = [discharge_note_df]
    else:
        return discharge_note_df


def query_radiology_note(hospital_stay, subject_id, timeline, engine):
    radiology_note_query = text(
        "select * from mimiciv.mimiciv_note.radiology where subject_id = :subject_id and charttime < :discharge_time"
    ).bindparams(subject_id=subject_id, discharge_time=timeline[-1]["time"])
    radiology_note_df = pd.read_sql(radiology_note_query, engine)

    if len(radiology_note_df) == 0:
        return

    # iterate through each note_id
    # for each note_id, get the radiology_detail
    radiology_note_details = pd.read_sql(
        text(
            "select * from mimiciv.mimiciv_note.radiology_detail where note_id in :note_id"
        ).bindparams(note_id=tuple(radiology_note_df["note_id"])),
        engine,
    )
    radiology_details_pivot = radiology_note_details.pivot_table(
        index=["note_id", "subject_id"],
        columns="field_name",
        values="field_value",
        aggfunc="first",
    ).reset_index()
    radiology_note_df = pd.merge(
        radiology_note_df,
        radiology_details_pivot,
        on=["note_id", "subject_id"],
        how="left",
    ).sort_values(by=["charttime"])
    radiology_note_df = radiology_note_df.drop(
        ["subject_id", "hadm_id", "note_seq", "storetime"], axis=1
    )
    # rename note_type based on a mapping
    # RR = Radiology Report
    # AR = Radiology Report Addendum
    radiology_note_df["note_type"] = radiology_note_df["note_type"].map(
        {"RR": "Radiology Report", "AR": "Radiology Report Addendum"}
    )
    column_name_mapping = {
        "note_id": "note id",
        "note_type": "type",
        "charttime": "chart time",
        "parent_note_id": "parent note id",
        "cpt_code": "cpt code",
        "exam_code": "exam code",
        "addendum_note_id": "addendum note id",
        "exam_name": "exam name",
    }
    # rename columns to be more readable
    radiology_note_df = radiology_note_df.rename(columns=column_name_mapping)
    hospital_stay["radiology notes"] = [radiology_note_df]


def upload_to_db(df, mimicllm_engine, table="data"):
    df.to_sql(
        table,
        mimicllm_engine,
        schema="mimicllm",
        if_exists="append",
        index=False,
        method="multi",
    )
