import pandas as pd
from sqlalchemy import text
import datetime

from src.processing.utils import (
    convert_lab_id_to_info,
    convert_icd_to_text,
    reorder_columns,
)
from src.database.queries import query_hosp, query_ed, query_discharge_note
from src.processing.timeline_generation import (
    get_time_intervals,
    generate_timeline,
    patient_timeline_to_text,
)


def post_process_hosp(hospital_stay, engine):
    for column in hospital_stay.columns:
        if (
            isinstance(hospital_stay[column][0], pd.DataFrame)
            and len(hospital_stay[column][0]) == 0
        ):
            continue

        match column:
            case "omr":
                omr_df = hospital_stay[column][0]

                omr_df = omr_df.rename(
                    columns={
                        "chartdate": "chart date",
                        "result_name": "name",
                        "result_value": "value",
                    }
                )

                hospital_stay[column] = [omr_df]
            case "diagnoses_icd":
                if len(hospital_stay[column][0]) > 0:
                    diagnoses_df = hospital_stay[column][0]

                    # remove any duplicate rows
                    diagnoses_df = diagnoses_df.drop_duplicates(
                        subset=["icd_code", "icd_version"]
                    )

                    ordered_diagnoses = convert_icd_to_text(
                        diagnoses_df, "diagnoses", engine
                    )

                    hospital_stay[column] = [ordered_diagnoses]
            case "labevents":
                converted = convert_lab_id_to_info(
                    hospital_stay[column][0], engine
                ).sort_values(by=["charttime"])

                converted = converted.rename(
                    columns={
                        "charttime": "chart time",
                        "valuenum": "numerical value",
                        "valueuom": "unit of measure",
                        "ref_range_lower": "normal lower end",
                        "ref_range_upper": "normal upper end",
                    }
                )

                converted = reorder_columns(
                    converted,
                    [
                        "chart time",
                        "category",
                        "fluid",
                        "label",
                        "priority",
                        "value",
                        "numerical value",
                        "unit of measure",
                        "normal lower end",
                        "normal upper end",
                        "flag",
                    ],
                )

                hospital_stay[column] = [converted]
            case "microbiologyevents":
                micro_df = hospital_stay[column][0]
                micro_df = micro_df.rename(
                    columns={
                        "charttime": "chart time",
                        "spec_type_desc": "specimen type",
                        "test_name": "test name",
                        "org_name": "organism name",
                        "isolate_num": "isolated colony",
                        "ab_name": "antibiotic name",
                        "dilution_text": "dilution",
                        "dilution_comparison": "dilution comparison",
                        "dilution_value": "dilution value",
                    }
                )
                hospital_stay[column] = [micro_df]
            case "patients":
                patients_df = hospital_stay[column][0]
                patients_df = patients_df.rename(
                    columns={
                        "anchor_age": "age at anchor year",
                        "anchor_year": "anchor year",
                        "dod": "date of death",
                    }
                )

                hospital_stay[column] = [patients_df]
            case "poe":
                poe_df = hospital_stay["poe"][0]
                poe_df = poe_df[
                    (poe_df["order_type"] != "ADT orders")
                    & (poe_df["order_type"] != "Lab")
                    & (poe_df["order_type"] != "Consult")
                    & (poe_df["order_type"] != "Blood Bank")
                ]

                # for each poe_id, check if there is a corresponding row in mimiciv_hosp.poe_detail (sql)
                # if there is, add the poe_detail to the poe row
                poe_ids = poe_df["poe_id"].tolist()
                poe_ids = [str(poe_id) for poe_id in poe_ids]

                if len(poe_ids) == 0:
                    hospital_stay["poe"] = [[]]
                    continue
                poe_detail_query = text(
                    f"select * from mimiciv.mimiciv_hosp.poe_detail where poe_id in :poe_ids"
                ).bindparams(poe_ids=tuple(poe_ids))
                poe_detail_df = pd.read_sql(poe_detail_query, engine).drop(
                    ["poe_seq", "subject_id"], axis=1
                )

                if len(poe_detail_df) > 0:
                    poe_df = poe_df.merge(poe_detail_df, on="poe_id", how="outer")

                    for i, row in poe_df.iterrows():
                        poe_df.at[i, row["field_name"]] = row["field_value"]

                    poe_df = poe_df.drop(["field_name", "field_value"], axis=1)

                hospital_stay["poe"] = [poe_df]
            case "prescriptions":
                poe_df = hospital_stay["poe"][0]
                prescriptions_df = hospital_stay["prescriptions"][0]

                if len(poe_df) > 0:
                    # remove any row if the order_type is Medicine and poe_id is not in prescriptions
                    poe_df = poe_df[
                        (poe_df["order_type"] != "Medications")
                        | (poe_df["poe_id"].isin(prescriptions_df["poe_id"]))
                    ]

                    # Get all the prescriptions that have a non-null value in the `poe_id` column
                    poe_prescriptions = prescriptions_df[
                        prescriptions_df["poe_id"].notnull()
                    ]
                    poe_prescriptions = poe_prescriptions.merge(
                        poe_df, on="poe_id", how="outer"
                    )

                    # if a row in poe_prescriptions has both a starttime and an ordertime, set ordertime as null
                    poe_prescriptions.loc[
                        (
                            (poe_prescriptions["starttime"].notnull())
                            & (poe_prescriptions["starttime"].notna())
                        )
                        & (
                            (poe_prescriptions["ordertime"].notnull())
                            & (poe_prescriptions["ordertime"].notna())
                        ),
                        "ordertime",
                    ] = None

                    # remove na
                    poe_prescriptions = poe_prescriptions.dropna(
                        subset=["starttime", "ordertime"]
                    )

                    poe_prescriptions["temp"] = poe_prescriptions[
                        "starttime"
                    ].combine_first(poe_prescriptions["ordertime"])

                    poe_prescriptions = poe_prescriptions.sort_values(by=["temp"])
                    poe_prescriptions = poe_prescriptions.drop("temp", axis=1)
                else:
                    poe_prescriptions = prescriptions_df.sort_values(by=["starttime"])

                poe_prescriptions = poe_prescriptions.rename(
                    columns={
                        "poe_id": "poe id",
                        "order_type": "order type",
                        "ordertime": "order time",
                        "transaction_type": "transaction type",
                        "order_subtype": "order subtype",
                        "drug_type": "drug type",
                        "prod_strength": "strength",
                        "dose_val_rx": "dose",
                        "dose_unit_rx": "dose unit",
                        "form_val_disp": "amount of medication",
                        "form_unit_disp": "amount unit",
                        "route": "route of administration",
                        "doses_per_24_hrs": "doses per 24 hours",
                        "discontinue_of_poe_id": "discontinues poe id",
                        "discontinued_by_poe_id": "discontinued by poe id",
                        "starttime": "start time",
                        "stoptime": "stop time",
                    }
                )

                hospital_stay["poe"] = [poe_prescriptions]

                hospital_stay = hospital_stay.drop("prescriptions", axis=1)
            case "procedures_icd":
                if len(hospital_stay[column][0]) > 0:
                    procedures_df = hospital_stay[column][0].copy()

                    # remove any duplicate rows
                    procedures_df = procedures_df.drop_duplicates(
                        subset=["icd_code", "icd_version"]
                    )

                    ordered_procedures = convert_icd_to_text(
                        procedures_df, "procedures", engine
                    )

                    procedures_df["name"] = ordered_procedures

                    procedures_df = procedures_df.drop(
                        ["icd_code", "icd_version"], axis=1
                    ).sort_values(by=["chartdate"])

                    procedures_df = procedures_df.rename(
                        columns={
                            "chartdate": "chart date",
                        }
                    )

                    hospital_stay[column] = [procedures_df]
            case "services":
                services_df = hospital_stay[column][0]
                services_df = services_df.rename(
                    columns={
                        "transfertime": "transfer time",
                        "curr_service": "current service",
                    }
                )
                hospital_stay[column] = [services_df]
            case "transfers":
                transfer_df = hospital_stay[column][0]
                transfer_df = transfer_df.sort_values(by=["intime"])
                transfer_df = transfer_df.rename(
                    columns={
                        "eventtype": "event type",
                        "careunit": "care unit",
                        "intime": "in time",
                        "outtime": "out time",
                    }
                )

                hospital_stay[column] = [transfer_df]
            case "race" | "marital_status" | "language":
                patients_df = hospital_stay["patients"][0].copy()
                patients_df[column] = hospital_stay[column]
                hospital_stay = hospital_stay.drop(column, axis=1)
                hospital_stay["patients"] = [patients_df]
    # rename columns to be more readable
    hospital_stay = hospital_stay.rename(
        columns={
            "admittime": "admission time",
            "dischtime": "discharge time",
            "deathtime": "death time",
            "admission_type": "admission type",
            "admission_location": "admission location",
            "discharge_location": "discharge location",
            "marital_status": "marital status",
            "diagnoses_icd": "diagnoses",
            "labevents": "lab tests",
            "microbiologyevents": "microbiology tests",
            "poe": "provider orders",
            "procedures_icd": "procedures",
            "services": "hospital services",
            "omr": "other patient information",
            "patients": "patient information",
        }
    )

    return hospital_stay


# get all info at and before a specific time
def filter_df_by_time(orig_df, time):
    df = orig_df.copy()

    for col in list(df.columns):
        cell = df[col][0]

        if isinstance(cell, pd.DataFrame):
            # find time columns
            time_cols = [
                col
                for col in cell.columns
                if col
                in [
                    "chart time",
                    "order time",
                    "start time",
                    "chart date",
                    "in time",
                    "transfer time",
                ]
            ]

            if len(time_cols) > 0 and len(cell) > 0:
                comparison = cell[time_cols[0]]

                if isinstance(comparison[0], datetime.date):
                    comparison = pd.to_datetime(comparison)
                # filter by time
                cell = cell[comparison <= time]

                df[col] = [cell]

            end_time_cols = [col for col in cell.columns if col in ["out time"]]

            if len(end_time_cols) > 0:
                # if the value for a cell in the end time column is after the time
                # set the value to null
                cell.loc[cell[end_time_cols[0]] > time, end_time_cols[0]] = None

        elif isinstance(cell, pd.Timestamp):
            if cell > time:
                df = df.drop(col, axis=1)

    if "discharge time" in orig_df.columns and orig_df["discharge time"][0] > time:
        df = df.drop("discharge location", axis=1)
    elif "exit time" in orig_df.columns and orig_df["exit time"][0] > time:
        df = df.drop("disposition", axis=1)

    return df


def filter_timeline_by_time(patient_info, timeline, time):
    filtered_patient_info = patient_info.copy()
    filtered_timeline = []

    filtered_patient_info = filtered_patient_info.drop("date of death", axis=1)

    for event in timeline:
        if event["time"] <= time:
            filtered_timeline.append(
                {
                    "type": event["type"],
                    "data": filter_df_by_time(event["data"], time),
                    "time": event["time"],
                }
            )

    return filtered_patient_info, filtered_timeline


def prompt_df_to_text(prompt_df, prompt):
    output = ""

    match prompt:
        case "lab tests":
            prompt_df = prompt_df.drop(
                [
                    "chart time",
                    "value",
                    "numerical value",
                    "unit of measure",
                    "normal lower end",
                    "normal upper end",
                    "flag",
                ],
                axis=1,
                errors="ignore",
            )
        case "microbiology tests":
            prompt_df = prompt_df.drop(
                [
                    "chart time",
                    "organism name",
                    "isolated colony",
                    "comments",
                    "interpretation",
                    "quantity",
                ],
                axis=1,
                errors="ignore",
            )
        case "provider orders":
            prompt_df = prompt_df.drop(
                [
                    "poe id",
                    "order time",
                    "start time",
                    "stop time",
                    "discontinues poe id",
                    "discontinued by poe id",
                ],
                axis=1,
                errors="ignore",
            )
        case "procedures":
            prompt_df = prompt_df.drop(["chart date"], axis=1, errors="ignore")

    if prompt == "diagnoses":
        output += f"Potential diagnoses:\n"
        for diagnosis in prompt_df:
            output += f" - {diagnosis}\n"
    else:
        output += f"Potential {prompt}:\n"
        for i, row in prompt_df.iterrows():
            output += f" - {i + 1} of {len(prompt_df)}\n"
            for column in prompt_df.columns:
                if str(row[column]).lower() not in ["none", "nan", "nat"]:
                    output += f"   - {column}: {row[column]}\n"

    return output


def generate_df_data(example_cases, example_prompts, subject_id, hadm_id):
    # map the prediction column as a user prompt
    prompt_to_text = {
        "lab tests": "What labs should be done next?",
        "microbiology tests": "What microbiology tests should be done next?",
        "provider orders": "What orders should be done next?",
        "procedures": "What procedures should be done next?",
        "diagnoses": "What diagnoses should be considered next?",
    }

    data = pd.DataFrame()

    seq_num = 0
    for i in range(len(example_cases)):
        patient_information = patient_timeline_to_text(
            example_cases[i]["patient_info"],
            example_cases[i]["timeline"],
        )

        user_prompts = list(example_prompts[i].keys())

        for prompt in user_prompts:
            sample_id = f"{hadm_id}_{seq_num}"
            seq_num += 1

            user_question = prompt_to_text[prompt]
            model_response = prompt_df_to_text(example_prompts[i][prompt], prompt)

            user_input = f"{patient_information}\n\n{user_question}"

            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        [
                            {
                                "sample_id": sample_id,
                                "subject_id": subject_id,
                                "hadm_id": hadm_id,
                                "input": user_input,
                                "output": model_response,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return data


def generate_sample_data(timeline, hospital_stay, subject_id, hadm_id):
    time_intervals = get_time_intervals(timeline)

    # finally, get the case at each time interval
    cases = []

    patient_info = hospital_stay["patient information"][0]
    hospital_diagnosis = hospital_stay["diagnoses"][0]

    for time in time_intervals:
        filtered_patient_info, filtered_timeline = filter_timeline_by_time(
            patient_info, timeline, time
        )

        cases.append(
            {
                "time": time,
                "patient_info": filtered_patient_info,
                "timeline": filtered_timeline,
            }
        )
    cols_to_predict = [
        "lab tests",
        "microbiology tests",
        "provider orders",
        "procedures",
        "diagnoses",
    ]

    prompts = []

    for i, case in enumerate(cases):
        prompts.append({})
        for j, event in enumerate(case["timeline"]):
            if event["type"] != "Hospital Stay":
                continue

            df = event["data"]
            for col in cols_to_predict:
                if col == "diagnoses":
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

                    next_timeline = next_case["timeline"]
                    next_hosp = next_timeline[j]["data"]

                    next_cell = next_hosp[col][0]

                    if not isinstance(next_cell, pd.DataFrame):
                        break

                    if not next_cell.equals(cell):
                        next_time = next_case["time"]
                        break

                # now we have the next time that col changes, find the differences between cell and next_cell
                if next_time is not None and next_cell is not None:
                    new_df = pd.concat([cell, next_cell]).drop_duplicates(keep=False)

                    prompts[i][col] = new_df.reset_index(drop=True)

    return generate_df_data(cases, prompts, subject_id, hadm_id)


def patient_info_to_sample(hadm_id, database_structure, engine):
    hospital_stay, subject_id = query_hosp(hadm_id, database_structure, engine)
    hospital_stay = post_process_hosp(hospital_stay, engine)

    ed_stays = query_ed(hadm_id, database_structure, engine)

    query_discharge_note(hadm_id, hospital_stay, engine)

    timeline = generate_timeline(hospital_stay, ed_stays, subject_id, engine)

    return generate_sample_data(timeline, hospital_stay, subject_id, hadm_id)
