from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os
import datetime
import pandas as pd
from sqlalchemy import text
from tqdm.auto import tqdm
from threading import Lock
import argparse
import ray
from sqlalchemy import Column, BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base

log_file_lock = Lock()

HOST_IP = os.environ['DATABASE_IP']
DATABASE_USER = os.environ['DATABASE_USER']
DATABASE_PASSWORD = os.environ['DATABASE_PASSWORD']
DATABASE_PORT = os.environ['DATABASE_PORT']

connection_url = URL.create(
    "postgresql+psycopg2",
    username=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=HOST_IP,
    port=int(DATABASE_PORT),
    database="mimiciv"
)

mimicllm_connection_url = URL.create(
    "postgresql+psycopg2",
    username=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=HOST_IP,
    port=int(DATABASE_PORT),
    database="mimicllm"
)

mimicllm_engine = create_engine(mimicllm_connection_url)

engine = create_engine(connection_url)

Base = declarative_base()

class Log(Base):
    __tablename__ = 'logs'
    __table_args__ = {'schema': 'mimicllm'}
    hadm_id = Column(BigInteger, primary_key=True)

    def __init__(self, hadm_id):
        self.hadm_id = hadm_id

query = text("""
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
""")

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


def to_clean_records(dataframe):
    return dataframe.apply(lambda row: row.dropna().to_dict(), axis=1).tolist()

def query_hosp(hadm_id):
    query = text("select * from mimiciv.mimiciv_hosp.admissions where hadm_id = :hadm;").bindparams(hadm=hadm_id)

    hospital_stay = pd.read_sql_query(query, engine)

    subject_id = int(hospital_stay['subject_id'][0])

    hosp_tables_to_ignore = [
        'admissions',
        'drgcodes',
        'emar',
        'hcpcsevents',
        'pharmacy'
    ]

    hosp_cols_to_remove = {
        'main': ['subject_id', 'hadm_id', 'admit_provider_id', 'insurance', 'edregtime', 'edouttime',
                 'hospital_expire_flag'],
        'diagnoses_icd': ['subject_id', 'hadm_id', 'seq_num'],
        'labevents': ['subject_id', 'hadm_id', 'labevent_id', 'order_provider_id', 'storetime', 'specimen_id'],
        'microbiologyevents': ['subject_id', 'hadm_id', 'order_provider_id', 'storetime', 'microevent_id',
                               'micro_specimen_id', 'spec_itemid', 'test_seq', 'test_itemid', 'chartdate', 'storedate',
                               'storetime', 'org_itemid', 'ab_itemid'],
        'poe': ['subject_id', 'hadm_id', 'order_provider_id', 'order_status', 'poe_seq'],
        'prescriptions': ['subject_id', 'hadm_id', 'pharmacy_id', 'order_provider_id', 'gsn', 'ndc',
                          'formulary_drug_cd',
                          'poe_seq'],
        'procedures_icd': ['subject_id', 'hadm_id', 'seq_num'],
        'services': ['subject_id', 'hadm_id'],
        'transfers': ['subject_id', 'hadm_id', 'transfer_id'],
        'omr': ['subject_id', 'seq_num'],
        'patients': ['subject_id', 'anchor_year_group']
    }

    for column, v in database_structure['mimiciv_hosp'].items():
        if 'hadm_id' in v:
            if column in hosp_tables_to_ignore:
                continue

            sql_query = text(f"select * from mimiciv.mimiciv_hosp.{column} where hadm_id = :hadm_id").bindparams(
                hadm_id=hadm_id)
            sql_df = pd.read_sql(sql_query, engine)

            removed_cols = hosp_cols_to_remove[column]
            sql_df = sql_df.drop(removed_cols, axis=1, errors='ignore')

            hospital_stay[column] = [sql_df]

    hospital_stay = hospital_stay.drop(
        hosp_cols_to_remove['main'], axis=1, errors='ignore'
    )

    # additional tables from subject id
    additional_patient_tables = [
        'omr',
        'patients'
    ]

    for table in additional_patient_tables:
        sql_query = text(f"select * from mimiciv.mimiciv_hosp.{table} where subject_id = :subject_id").bindparams(
            subject_id=subject_id)
        sql_df = pd.read_sql(sql_query, engine)
        removed_cols = hosp_cols_to_remove[table]
        sql_df = sql_df.drop(removed_cols, axis=1, errors='ignore')

        hospital_stay[table] = [sql_df]

    return hospital_stay, subject_id


def convert_icd_to_text(icd_list, icd_type):
    # Prepare a CASE statement for ordering
    case_statement = "CASE "
    for index, item in icd_list.iterrows():
        code = item['icd_code'].strip()  # Remove leading/trailing spaces
        version = item['icd_version']
        case_statement += f"WHEN icd_code = '{code}' AND icd_version = {version} THEN {index} "

    case_statement += "END"

    # Create a WHERE IN clause
    icd_conditions = ", ".join(
        [f"('{item[1]['icd_code'].strip()}', {item[1]['icd_version']})" for item in icd_list.iterrows()])
    sql_query = f"""
    SELECT long_title 
    FROM mimiciv.mimiciv_hosp.d_icd_{icd_type} 
    WHERE (icd_code, icd_version) IN ({icd_conditions})
    ORDER BY {case_statement};
    """

    # Execute the query
    return pd.read_sql(sql_query, engine)['long_title'].tolist()


def convert_lab_id_to_info(labs):
    # Prepare a CASE statement for ordering
    case_statement = "CASE "
    for index, item in labs.iterrows():
        id = item['itemid']
        case_statement += f"WHEN itemid = {id} THEN {index} "

    case_statement += "END"

    # Create a WHERE IN clause
    lab_conditions = ", ".join([str(item[1]['itemid']) for item in labs.iterrows()])
    sql_query = f"""
    SELECT *
    FROM mimiciv.mimiciv_hosp.d_labitems
    WHERE itemid IN ({lab_conditions})
    ORDER BY {case_statement};
    """

    # Execute the query
    returned = pd.read_sql(sql_query, engine)

    return labs.merge(returned, on='itemid', how='outer').drop('itemid', axis=1)


def reorder_columns(df, columns):
    return df[[column for column in columns if column in df.columns]]


def post_process_hosp(hospital_stay):
    for column in hospital_stay.columns:
        if isinstance(hospital_stay[column][0], pd.DataFrame) and len(hospital_stay[column][0]) == 0:
            continue

        match column:
            case 'omr':
                omr_df = hospital_stay[column][0]

                omr_df = omr_df.rename(columns={
                    'chartdate': 'chart date',
                    'result_name': 'name',
                    'result_value': 'value',
                })

                hospital_stay[column] = [omr_df]
            case 'diagnoses_icd':
                if len(hospital_stay[column][0]) > 0:
                    diagnoses_df = hospital_stay[column][0]
                    
                    # remove any duplicate rows
                    diagnoses_df = diagnoses_df.drop_duplicates(subset=['icd_code', 'icd_version'])
                    
                    ordered_diagnoses = convert_icd_to_text(diagnoses_df, 'diagnoses')

                    hospital_stay[column] = [ordered_diagnoses]
            case 'labevents':
                converted = convert_lab_id_to_info(hospital_stay[column][0]).sort_values(by=['charttime'])

                converted = converted.rename(columns={
                    'charttime': 'chart time',
                    'valuenum': 'numerical value',
                    'valueuom': 'unit of measure',
                    'ref_range_lower': 'normal lower end',
                    'ref_range_upper': 'normal upper end',
                })

                converted = reorder_columns(converted, [
                    'chart time', 'category', 'fluid',
                    'label', 'priority', 'value', 'numerical value',
                    'unit of measure', 'normal lower end', 'normal upper end',
                    'flag'
                ])

                hospital_stay[column] = [converted]
            case 'microbiologyevents':
                micro_df = hospital_stay[column][0]
                micro_df = micro_df.rename(columns={
                    'charttime': 'chart time',
                    'spec_type_desc': 'specimen type',
                    'test_name': 'test name',
                    'org_name': 'organism name',
                    'isolate_num': 'isolated colony',
                    'ab_name': 'antibiotic name',
                    'dilution_text': 'dilution',
                    'dilution_comparison': 'dilution comparison',
                    'dilution_value': 'dilution value',
                })
                hospital_stay[column] = [micro_df]
            case 'patients':
                patients_df = hospital_stay[column][0]
                patients_df = patients_df.rename(columns={
                    'anchor_age': 'age at anchor year',
                    'anchor_year': 'anchor year',
                    'dod': 'date of death',
                })

                hospital_stay[column] = [patients_df]
            case 'poe':
                poe_df = hospital_stay['poe'][0]
                poe_df = poe_df[
                    (poe_df['order_type'] != 'ADT orders') &
                    (poe_df['order_type'] != 'Lab') &
                    (poe_df['order_type'] != 'Consult') &
                    (poe_df['order_type'] != 'Blood Bank')
                ]

                # for each poe_id, check if there is a corresponding row in mimiciv_hosp.poe_detail (sql)
                # if there is, add the poe_detail to the poe row
                poe_ids = poe_df['poe_id'].tolist()
                poe_ids = [str(poe_id) for poe_id in poe_ids]

                if len(poe_ids) == 0:
                    hospital_stay['poe'] = []
                    continue
                poe_detail_query = text(
                    f"select * from mimiciv.mimiciv_hosp.poe_detail where poe_id in :poe_ids").bindparams(poe_ids=tuple(poe_ids))
                poe_detail_df = pd.read_sql(poe_detail_query, engine).drop(['poe_seq', 'subject_id'], axis=1)

                if len(poe_detail_df) > 0:
                    poe_df = poe_df.merge(poe_detail_df, on='poe_id', how='outer')

                    for i, row in poe_df.iterrows():
                        poe_df.at[i, row['field_name']] = row['field_value']

                    poe_df = poe_df.drop(['field_name', 'field_value'], axis=1)

                hospital_stay['poe'] = [poe_df]
            case 'prescriptions':
                poe_df = hospital_stay['poe'][0]
                prescriptions_df = hospital_stay['prescriptions'][0]

                # remove any row if the order_type is Medicine and poe_id is not in prescriptions
                poe_df = poe_df[
                    (poe_df['order_type'] != 'Medications') |
                    (poe_df['poe_id'].isin(prescriptions_df['poe_id']))
                    ]

                # Get all the prescriptions that have a non-null value in the `poe_id` column
                poe_prescriptions = prescriptions_df[prescriptions_df['poe_id'].notnull()]
                poe_prescriptions = poe_prescriptions.merge(poe_df, on='poe_id', how='outer')

                # if a row in poe_prescriptions has both a starttime and an ordertime, set ordertime as null
                poe_prescriptions.loc[(poe_prescriptions['starttime'].notnull()) & (
                    poe_prescriptions['ordertime'].notnull()), 'ordertime'] = None

                poe_prescriptions['temp'] = poe_prescriptions['starttime'].combine_first(poe_prescriptions['ordertime'])

                poe_prescriptions = poe_prescriptions.sort_values(by=['temp'])
                poe_prescriptions = poe_prescriptions.drop('temp', axis=1)

                poe_prescriptions = poe_prescriptions.rename(columns={
                    'poe_id': 'poe id',
                    'order_type': 'order type',
                    'ordertime': 'order time',
                    'transaction_type': 'transaction type',
                    'order_subtype': 'order subtype',
                    'drug_type': 'drug type',
                    'prod_strength': 'strength',
                    'dose_val_rx': 'dose',
                    'dose_unit_rx': 'dose unit',
                    'form_val_disp': 'amount of medication',
                    'form_unit_disp': 'amount unit',
                    'route': 'route of administration',
                    'doses_per_24_hrs': 'doses per 24 hours',
                    'discontinue_of_poe_id': 'discontinues poe id',
                    'discontinued_by_poe_id': 'discontinued by poe id',
                    'starttime': 'start time',
                    'stoptime': 'stop time',
                })

                hospital_stay['poe'] = [poe_prescriptions]

                hospital_stay = hospital_stay.drop('prescriptions', axis=1)
            case 'procedures_icd':
                if len(hospital_stay[column][0]) > 0:
                    procedures_df = hospital_stay[column][0].copy()

                    # remove any duplicate rows
                    procedures_df = procedures_df.drop_duplicates(subset=['icd_code', 'icd_version'])

                    ordered_procedures = convert_icd_to_text(procedures_df, 'procedures')
                    
                    procedures_df['name'] = ordered_procedures

                    procedures_df = procedures_df.drop(['icd_code', 'icd_version'], axis=1).sort_values(
                        by=['chartdate'])

                    procedures_df = procedures_df.rename(columns={
                        'chartdate': 'chart date',
                    })

                    hospital_stay[column] = [procedures_df]
            case 'services':
                services_df = hospital_stay[column][0]
                services_df = services_df.rename(columns={
                    'transfertime': 'transfer time',
                    'curr_service': 'current service',
                })
                hospital_stay[column] = [services_df]
            case 'transfers':
                transfer_df = hospital_stay[column][0]
                transfer_df = transfer_df.sort_values(by=['intime'])
                transfer_df = transfer_df.rename(columns={
                    'eventtype': 'event type',
                    'careunit': 'care unit',
                    'intime': 'in time',
                    'outtime': 'out time',
                })

                hospital_stay[column] = [transfer_df]
            case 'race' | 'marital_status' | 'language':
                patients_df = hospital_stay['patients'][0].copy()
                patients_df[column] = hospital_stay[column]
                hospital_stay = hospital_stay.drop(column, axis=1)
                hospital_stay['patients'] = [patients_df]
    # rename columns to be more readable
    hospital_stay = hospital_stay.rename(columns={
        'admittime': 'admission time',
        'dischtime': 'discharge time',
        'deathtime': 'death time',
        'admission_type': 'admission type',
        'admission_location': 'admission location',
        'discharge_location': 'discharge location',
        'marital_status': 'marital status',
        'diagnoses_icd': 'diagnoses',
        'labevents': 'lab tests',
        'microbiologyevents': 'microbiology tests',
        'poe': 'provider orders',
        'procedures_icd': 'procedures',
        'services': 'hospital services',
        'omr': 'other patient information',
        'patients': 'patient information'
    })

    return hospital_stay


def query_ed(hadm_id):
    ed_stay_query = text("select * from mimiciv.mimiciv_ed.edstays where hadm_id = :hadm_id").bindparams(
        hadm_id=hadm_id)

    # stay_id = 32522732
    # ed_stay_query = text("select * from mimiciv.mimiciv_ed.edstays where stay_id = :stay_id").bindparams(stay_id=stay_id)

    ed_stay_df = pd.read_sql(ed_stay_query, engine)

    ed_stays = []

    ed_cols_to_remove = {
        'main': ['subject_id', 'race', 'gender', 'stay_id'],
        'diagnosis': ['subject_id', 'stay_id', 'seq_num', 'icd_code', 'icd_version'],
        'medrecon': ['subject_id', 'stay_id', 'gsn', 'ndc', 'etc_rn', 'etccode', 'gsn'],
        'pyxis': ['subject_id', 'stay_id', 'gsn', 'med_rn', 'gsn', 'gsn_rn'],
        'triage': ['subject_id', 'stay_id'],
        'vitalsign': ['subject_id', 'stay_id']
    }

    for row in ed_stay_df.iterrows():
        row_info = pd.DataFrame(row[1]).transpose()
        stay_id = row_info['stay_id'].values[0]

        for column, value in database_structure['mimiciv_ed'].items():
            if 'stay_id' in value:
                if column == 'edstays':
                    continue

                sql_query = text(f"select * from mimiciv.mimiciv_ed.{column} where stay_id = :stay_id").bindparams(
                    stay_id=stay_id)
                sql_df = pd.read_sql(sql_query, engine)

                removed_cols = ed_cols_to_remove[column]
                sql_df = sql_df.drop(removed_cols, axis=1, errors='ignore')

                match column:
                    case 'triage':
                        # reorder and rename columns
                        sql_df = sql_df.rename(columns={
                            'heartrate': 'heart rate',
                            'resprate': 'respiratory rate',
                            'o2sat': 'oxygen saturation',
                            'sbp': 'systolic blood pressure',
                            'dbp': 'diastolic blood pressure',
                            'chiefcomplaint': 'chief complaint',
                            'acuity': 'emergency severity index'
                        })

                        sql_df = reorder_columns(sql_df, [
                            'chief complaint', 'pain', 'emergency severity index',
                            'temperature', 'heart rate', 'respiratory rate',
                            'oxygen saturation', 'systolic blood pressure', 'diastolic blood pressure',
                        ])
                    case 'vitalsign':
                        # rename columns
                        sql_df = sql_df.rename(columns={
                            'charttime': 'chart time',
                            'heartrate': 'heart rate',
                            'resprate': 'respiratory rate',
                            'o2sat': 'oxygen saturation',
                            'sbp': 'systolic blood pressure',
                            'dbp': 'diastolic blood pressure',
                        })
                    case 'diagnosis':
                        # convert to a list of icd_title column
                        sql_df = sql_df['icd_title'].tolist()
                    case 'medrecon':
                        sql_df = sql_df.drop('charttime', axis=1, errors='ignore')
                        sql_df = sql_df.rename(columns={'etcdescription': 'enhanced therapeutic class'})
                    case 'pyxis':
                        # remove duplicate rows if `name` AND `charttime` are the same
                        sql_df = sql_df.drop_duplicates(subset=['name', 'charttime'])
                        sql_df = sql_df.rename(columns={
                            'charttime': 'chart time',
                        })

                # if isinstance(sql_df, pd.DataFrame):
                #     sql_df = to_clean_records(sql_df)

                row_info[column] = [sql_df]
        ed_stays.append(row_info)

    ed_stays = pd.concat(ed_stays, ignore_index=True)

    ed_stays = ed_stays.drop(
        ed_cols_to_remove['main'], axis=1, errors='ignore'
    )

    # rename columns to be more readable
    ed_stays = ed_stays.rename(columns={
        'intime': 'arrival time',
        'outtime': 'exit time',
        'arrival_transport': 'arrival transport',
        'medrecon': 'medication reconciliation',
        'pyxis': 'dispensed medications',
        'vitalsign': 'vital signs'
    })

    # reorder columns
    ed_stays = reorder_columns(ed_stays, [
        'arrival time', 'arrival transport',
        'triage',
        'vital signs',
        'medication reconciliation',
        'dispensed medications',
        'exit time', 'disposition',
        'diagnosis'
    ])

    return ed_stays


def query_discharge_note(hadm_id, hospital_stay):
    discharge_note_query = text("select * from mimiciv.mimiciv_note.discharge where hadm_id = :hadm_id").bindparams(
        hadm_id=hadm_id)

    discharge_note_df = pd.read_sql(discharge_note_query, engine)

    discharge_note_df = discharge_note_df.drop(
        ['note_id', 'subject_id', 'hadm_id', 'note_type', 'note_seq', 'charttime'],
        axis=1
    )

    discharge_note_df = discharge_note_df.rename(columns={
        'storetime': 'chart time',
        'text': 'discharge note'
    })

    hospital_stay['discharge note'] = [discharge_note_df]


def generate_timeline(hospital_stay, ed_stays, subject_id):
    timeline = []

    # add hospital stay
    timeline.append({
        'type': 'Hospital Stay',
        'data': hospital_stay,
        'time': hospital_stay['admission time'][0]
    })

    # add ed stays
    for i, row in ed_stays.iterrows():
        timeline.append({
            'type': 'Emergency Department Stay',
            'data': pd.DataFrame([row]).reset_index(drop=True),
            'time': row['arrival time']
        })

    # sort timeline by time
    timeline = sorted(timeline, key=lambda k: k['time'])
    query_radiology_note(hospital_stay, subject_id, timeline)

    col_order = [
        'patient information',
        'race', 'marital status', 'language',
        'admission time', 'admission type', 'admission location',
        'hospital services', 'transfers',
        'lab tests', 'microbiology tests', 'procedures',
        'provider orders',
        'other patient information',
        'discharge time', 'discharge location', 'death time',
        'radiology notes', 'discharge note',
        'diagnoses'
    ]

    ed_medrecon = None

    for i, event in enumerate(timeline):
        if event['type'] == 'Hospital Stay':
            event['data'] = reorder_columns(event['data'], col_order)
        elif event['type'] == 'Emergency Department Stay':
            if ed_medrecon is None:
                ed_medrecon = event['data']['medication reconciliation'][0]
            else:
                current_medrecon = event['data']['medication reconciliation'][0]
                # only keep the rows that are not in ed_medrecon
                current_medrecon = current_medrecon[~current_medrecon['name'].isin(ed_medrecon['name'])]

                event['data']['medication reconciliation'] = [current_medrecon]
                ed_medrecon = pd.concat([ed_medrecon, current_medrecon], ignore_index=True)

    return timeline


def query_radiology_note(hospital_stay, subject_id, timeline):
    radiology_note_query = text(
        "select * from mimiciv.mimiciv_note.radiology where subject_id = :subject_id and charttime < :discharge_time").bindparams(
        subject_id=subject_id, discharge_time=timeline[-1]['time'])
    radiology_note_df = pd.read_sql(radiology_note_query, engine)

    if len(radiology_note_df) == 0:
        return

    # iterate through each note_id
    # for each note_id, get the radiology_detail
    radiology_note_details = pd.read_sql(
        text("select * from mimiciv.mimiciv_note.radiology_detail where note_id in :note_id").bindparams(
            note_id=tuple(radiology_note_df['note_id'])), engine)
    radiology_details_pivot = radiology_note_details.pivot_table(
        index=['note_id', 'subject_id'],
        columns='field_name',
        values='field_value',
        aggfunc='first'
    ).reset_index()
    radiology_note_df = pd.merge(
        radiology_note_df,
        radiology_details_pivot,
        on=['note_id', 'subject_id'],
        how='left'
    ).sort_values(by=['charttime'])
    radiology_note_df = radiology_note_df.drop(
        ['subject_id', 'hadm_id', 'note_seq', 'storetime'],
        axis=1
    )
    # rename note_type based on a mapping
    # RR = Radiology Report
    # AR = Radiology Report Addendum
    radiology_note_df['note_type'] = radiology_note_df['note_type'].map({
        'RR': 'Radiology Report',
        'AR': 'Radiology Report Addendum'
    })
    column_name_mapping = {
        'note_id': 'note id',
        'note_type': 'type',
        'charttime': 'chart time',
        'parent_note_id': 'parent note id',
        'cpt_code': 'cpt code',
        'exam_code': 'exam code',
        'addendum_note_id': 'addendum note id',
        'exam_name': 'exam name',
    }
    # rename columns to be more readable
    radiology_note_df = radiology_note_df.rename(columns=column_name_mapping)
    hospital_stay['radiology notes'] = [radiology_note_df]

def format_df_to_text(df):
    output = ""

    for row in df.iterrows():
        for column in df.columns:
            if column in ['subject_id', 'hadm_id']:
                continue

            value = row[1][column]

            if isinstance(value, pd.DataFrame):
                value = to_clean_records(value)

            if isinstance(value, list) and len(value) > 0:
                output += f"{column}:\n"

                if isinstance(value[0], dict):
                    for i, dictionary in enumerate(value):
                        output += f" - {i + 1} of {len(value)}\n"
                        for k, v in dictionary.items():
                            if str(v).lower() not in ['none', 'nan', 'nat']:
                                output += f"   - {k}: {v}\n"
                else:
                    for v in value:
                        if str(v).lower() not in ['none', 'nan', 'nat']:
                            output += f" - {v}\n"
            elif not isinstance(value, list) and str(value).lower() not in ['none', 'nan', 'nat']:
                output += f"{column}: {value}\n"

        output += "\n"

    return output


def patient_timeline_to_text(patient_preamble, timeline, hospital_diagnosis=None):
    patient_info = "Patient Information:\n"

    patient_info += format_df_to_text(patient_preamble) + "\n"

    for i, event in enumerate(timeline):
        if event['type'] == 'Hospital Stay':
            event['data'] = event['data'].drop(['patient information', 'diagnoses'], axis=1, errors='ignore')

        patient_info += f"{event['type']} ({i + 1} of {len(timeline)})\n"
        patient_info += format_df_to_text(event['data'])
        patient_info += "\n"

    if hospital_diagnosis:
        patient_info += "Diagnoses:\n"
        for diagnosis in hospital_diagnosis:
            patient_info += f" - {diagnosis}\n"

    return patient_info


# get all info at and before a specific time
def filter_df_by_time(orig_df, time):
    df = orig_df.copy()

    for col in list(df.columns):
        cell = df[col][0]

        if isinstance(cell, pd.DataFrame):
            # find time columns
            time_cols = [col for col in cell.columns if col in [
                'chart time', 'order time', 'start time', 'chart date', 'in time', 'transfer time'
            ]]

            if len(time_cols) > 0 and len(cell) > 0:
                comparison = cell[time_cols[0]]

                if isinstance(comparison[0], datetime.date):
                    comparison = pd.to_datetime(comparison)
                # filter by time
                cell = cell[comparison <= time]

                df[col] = [cell]

            end_time_cols = [col for col in cell.columns if col in [
                'out time'
            ]]

            if len(end_time_cols) > 0:
                # if the value for a cell in the end time column is after the time
                # set the value to null
                cell.loc[cell[end_time_cols[0]] > time, end_time_cols[0]] = None

        elif isinstance(cell, pd.Timestamp):
            if cell > time:
                df = df.drop(col, axis=1)

    if 'discharge time' in orig_df.columns and orig_df['discharge time'][0] > time:
        df = df.drop('discharge location', axis=1)
    elif 'exit time' in orig_df.columns and orig_df['exit time'][0] > time:
        df = df.drop('disposition', axis=1)

    return df


def filter_timeline_by_time(patient_info, timeline, time):
    filtered_patient_info = patient_info.copy()
    filtered_timeline = []

    filtered_patient_info = filtered_patient_info.drop('date of death', axis=1)

    for event in timeline:
        if event['time'] <= time:
            filtered_timeline.append({
                'type': event['type'],
                'data': filter_df_by_time(event['data'], time),
                'time': event['time']
            })

    return filtered_patient_info, filtered_timeline

def get_time_intervals(timeline):
    time_columns_to_include = [
        'vital signs', 'transfers', 'lab tests', 'microbiology tests', 'procedures', 'provider orders',
        'radiology notes',
        'discharge note'
    ]

    time_intervals = set()

    lower_bound = timeline[0]['time']

    for event in timeline:
        if event['type'] != 'Hospital Stay':
            continue

        lower_bound = event['time']
        time_intervals.add(event['time'])

        for col in event['data'].columns:
            cell = event['data'][col][0]

            if isinstance(cell, pd.DataFrame):
                if col not in time_columns_to_include:
                    continue
                # find time columns
                time_cols = [col for col in cell.columns if col in [
                    'chart time', 'order time', 'start time', 'chart date', 'in time', 'transfer time'
                ]]

                if len(time_cols) > 0 and len(cell) > 0:
                    comparison = cell[time_cols[0]]

                    if isinstance(comparison[0], datetime.date):
                        comparison = pd.to_datetime(comparison)

                    time_intervals.update(comparison)

                end_time_cols = [col for col in cell.columns if col in [
                    'out time'
                ]]

                if len(end_time_cols) > 0:
                    time_intervals.update(cell[end_time_cols[0]])
            elif isinstance(cell, pd.Timestamp):
                time_intervals.add(cell)

    time_intervals = [time for time in time_intervals if time >= lower_bound and pd.isna(time) == False]

    return sorted(time_intervals)


def prompt_df_to_text(prompt_df, prompt):
    output = ""

    match prompt:
        case 'lab tests':
            prompt_df = prompt_df.drop([
                'chart time', 'value', 'numerical value', 'unit of measure', 'normal lower end', 'normal upper end',
                'flag'
            ], axis=1, errors='ignore')
        case 'microbiology tests':
            prompt_df = prompt_df.drop([
                'chart time', 'organism name', 'isolated colony', 'comments', 'interpretation', 'quantity'
            ], axis=1, errors='ignore')
        case 'provider orders':
            prompt_df = prompt_df.drop([
                'poe id', 'order time', 'start time', 'stop time', 'discontinues poe id', 'discontinued by poe id'
            ], axis=1, errors='ignore')
        case 'procedures':
            prompt_df = prompt_df.drop([
                'chart date'
            ], axis=1, errors='ignore')

    if prompt == 'diagnoses':
        output += f"Potential diagnoses:\n"
        for diagnosis in prompt_df:
            output += f" - {diagnosis}\n"
    else:
        output += f"Potential {prompt}:\n"
        for i, row in prompt_df.iterrows():
            output += f" - {i + 1} of {len(prompt_df)}\n"
            for column in prompt_df.columns:
                if str(row[column]).lower() not in ['none', 'nan', 'nat']:
                    output += f"   - {column}: {row[column]}\n"

    return output


def generate_df_data(example_cases, example_prompts, subject_id, hadm_id):
    # map the prediction column as a user prompt
    prompt_to_text = {
        'lab tests': 'What labs should be done next?',
        'microbiology tests': 'What microbiology tests should be done next?',
        'provider orders': 'What orders should be done next?',
        'procedures': 'What procedures should be done next?',
        'diagnoses': 'What diagnoses should be considered next?'
    }

    data = pd.DataFrame()

    seq_num = 0
    for i in range(len(example_cases)):
        patient_information = patient_timeline_to_text(
            example_cases[i]['patient_info'],
            example_cases[i]['timeline'],
        )

        user_prompts = list(example_prompts[i].keys())

        for prompt in user_prompts:
            sample_id = f"{hadm_id}_{seq_num}"
            seq_num += 1

            user_question = prompt_to_text[prompt]
            model_response = prompt_df_to_text(example_prompts[i][prompt], prompt)

            user_input = f"{patient_information}\n\n{user_question}"

            data = pd.concat([
                data,
                pd.DataFrame([{
                    'sample_id': sample_id,
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'input': user_input,
                    'output': model_response
                }])
            ], ignore_index=True)

    return data


def generate_sample_data(timeline, hospital_stay, subject_id, hadm_id):
    time_intervals = get_time_intervals(timeline)

    # finally, get the case at each time interval
    cases = []

    patient_info = hospital_stay['patient information'][0]
    hospital_diagnosis = hospital_stay['diagnoses'][0]

    for time in time_intervals:
        filtered_patient_info, filtered_timeline = filter_timeline_by_time(patient_info, timeline, time)

        cases.append({
            'time': time,
            'patient_info': filtered_patient_info,
            'timeline': filtered_timeline
        })
    cols_to_predict = [
        'lab tests', 'microbiology tests', 'provider orders', 'procedures', 'diagnoses'
    ]

    prompts = []

    for i, case in enumerate(cases):
        prompts.append({})
        for j, event in enumerate(case['timeline']):
            if event['type'] != 'Hospital Stay':
                continue

            df = event['data']
            for col in cols_to_predict:
                if col == 'diagnoses':
                    prompts[i][col] = hospital_diagnosis
                    continue

                if col not in df.columns:
                    continue

                # find the value of col
                cell = df[col][0]

                # now iterate through the cases after the current one until there is a change in col
                next_time = None
                next_cell = None
                for k in range(i + 1, len(cases)):
                    next_case = cases[k]

                    next_timeline = next_case['timeline']
                    next_hosp = next_timeline[j]['data']

                    next_cell = next_hosp[col][0]

                    if not next_cell.equals(cell):
                        next_time = next_case['time']
                        break

                # now we have the next time that col changes, find the differences between cell and next_cell
                if next_time is not None and next_cell is not None:
                    new_df = pd.concat([cell, next_cell]).drop_duplicates(keep=False)

                    prompts[i][col] = new_df.reset_index(drop=True)

    return generate_df_data(cases, prompts, subject_id, hadm_id)

def patient_info_to_sample(hadm_id):
    hospital_stay, subject_id = query_hosp(hadm_id)
    hospital_stay = post_process_hosp(hospital_stay)

    ed_stays = query_ed(hadm_id)

    query_discharge_note(hadm_id, hospital_stay)

    timeline = generate_timeline(hospital_stay, ed_stays, subject_id)

    return generate_sample_data(timeline, hospital_stay, subject_id, hadm_id)

def fetch_all_hadm_ids():
    query = text("""
    SELECT a.hadm_id
    FROM mimiciv_hosp.admissions a
    INNER JOIN mimiciv_ed.edstays e ON a.hadm_id = e.hadm_id
    WHERE EXTRACT(day FROM (a.dischtime - a.admittime)) < 7;
    """)

    with engine.connect() as connection:
        result = connection.execute(query).fetchall()

    list_results = [item[0] for item in result]

    return list_results

def upload_to_db(df):
    df.to_sql('data', mimicllm_engine, schema='mimicllm', if_exists='append', index=False, method='multi')

def read_processed_hadm_ids(rewrite=False):
    Session = sessionmaker(bind=mimicllm_engine)
    session = Session()

    if rewrite:
        # Use TRUNCATE to efficiently clear the table
        session.execute("TRUNCATE TABLE mimicllm.logs;")
        session.commit()

    try:
        # Query the database for all hadm_id values
        processed_hadm_ids = {log.hadm_id for log in session.query(Log.hadm_id).all()}
        return processed_hadm_ids
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def log_hadm_id(hadm_id):
    Session = sessionmaker(bind=mimicllm_engine)
    session = Session()

    new_log = Log(hadm_id=hadm_id)
    session.add(new_log)

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

@ray.remote
def process_hadm_id(hadm_id):
    try:
        df = patient_info_to_sample(hadm_id)
        upload_to_db(df)
        log_hadm_id(hadm_id)  # Log the processed hadm_id
        return hadm_id, None
    except Exception as e:
        error_message = f"Error processing hadm_id {hadm_id}: {type(e).__name__} errored with message: {e}"
        return hadm_id, error_message

def main():
    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Process MIMIC-IV data into a format that can be used by an LLM")
    parser.add_argument('--rewrite-log-db', action='store_true',
                        help='If set, will rewrite the log database')

    args = parser.parse_args()

    rewrite_log_file = args.rewrite_log_file

    processed_hadm_ids = read_processed_hadm_ids(rewrite=rewrite_log_file)
    all_hadm_ids = fetch_all_hadm_ids()
    hadm_ids = [hadm_id for hadm_id in all_hadm_ids if hadm_id not in processed_hadm_ids]

    context = ray.init()
    print(f"Connected to Ray Dashboard at {context.dashboard_url}")

    # Set up the progress bar
    with tqdm(total=len(hadm_ids), desc='Processing', dynamic_ncols=True) as pbar:
        futures = [process_hadm_id.remote(hadm_id) for hadm_id in hadm_ids]

        for future in ray.get(futures):
            hadm_id, error = future
            if error:
                print(error)  # Ray takes care of orderly printing
            pbar.set_description(f"Completed hadm_id {hadm_id}")
            pbar.update(1)


if __name__ == '__main__':
    main()