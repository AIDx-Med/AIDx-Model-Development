import pandas as pd
import datetime

from src.database.queries import query_radiology_note
from src.processing.utils import reorder_columns, to_clean_records


def generate_timeline(hospital_stay, ed_stays, subject_id, engine):
    # add hospital stay
    timeline = [
        {
            "type": "Hospital Stay",
            "data": hospital_stay,
            "time": hospital_stay["admission time"][0],
        }
    ]

    # add ed stays
    for i, row in ed_stays.iterrows():
        timeline.append(
            {
                "type": "Emergency Department Stay",
                "data": pd.DataFrame([row]).reset_index(drop=True),
                "time": row["arrival time"],
            }
        )

    # sort timeline by time
    timeline = sorted(timeline, key=lambda k: k["time"])
    query_radiology_note(hospital_stay, subject_id, timeline, engine)

    col_order = [
        "patient information",
        "race",
        "marital status",
        "language",
        "admission time",
        "admission type",
        "admission location",
        "hospital services",
        "transfers",
        "lab tests",
        "microbiology tests",
        "procedures",
        "provider orders",
        "other patient information",
        "discharge time",
        "discharge location",
        "death time",
        "radiology notes",
        "discharge note",
        "diagnoses",
    ]

    ed_medrecon = None

    for i, event in enumerate(timeline):
        if event["type"] == "Hospital Stay":
            event["data"] = reorder_columns(event["data"], col_order)
        elif event["type"] == "Emergency Department Stay":
            if ed_medrecon is None:
                ed_medrecon = event["data"]["medication reconciliation"][0]
            else:
                current_medrecon = event["data"]["medication reconciliation"][0]
                # only keep the rows that are not in ed_medrecon
                current_medrecon = current_medrecon[
                    ~current_medrecon["name"].isin(ed_medrecon["name"])
                ]

                event["data"]["medication reconciliation"] = [current_medrecon]
                ed_medrecon = pd.concat(
                    [ed_medrecon, current_medrecon], ignore_index=True
                )

    return timeline


def patient_timeline_to_text(patient_preamble, timeline, hospital_diagnosis=None):
    patient_info = "Patient Information:\n"

    patient_info += format_df_to_text(patient_preamble) + "\n"

    for i, event in enumerate(timeline):
        if event["type"] == "Hospital Stay":
            event["data"] = event["data"].drop(
                ["patient information", "diagnoses"], axis=1, errors="ignore"
            )

        patient_info += f"{event['type']} ({i + 1} of {len(timeline)})\n"
        patient_info += format_df_to_text(event["data"])
        patient_info += "\n"

    if hospital_diagnosis:
        patient_info += "Diagnoses:\n"
        for diagnosis in hospital_diagnosis:
            patient_info += f" - {diagnosis}\n"

    return patient_info


def get_time_intervals(timeline):
    time_columns_to_include = [
        "vital signs",
        "transfers",
        "lab tests",
        "microbiology tests",
        "procedures",
        "provider orders",
        "radiology notes",
        "discharge note",
    ]

    time_intervals = set()

    lower_bound = timeline[0]["time"]

    for event in timeline:
        if event["type"] != "Hospital Stay":
            continue

        lower_bound = event["time"]
        time_intervals.add(event["time"])

        for col in event["data"].columns:
            cell = event["data"][col][0]

            if isinstance(cell, pd.DataFrame):
                if col not in time_columns_to_include:
                    continue
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

                    time_intervals.update(comparison)

                end_time_cols = [col for col in cell.columns if col in ["out time"]]

                if len(end_time_cols) > 0:
                    time_intervals.update(cell[end_time_cols[0]])
            elif isinstance(cell, pd.Timestamp):
                time_intervals.add(cell)

    time_intervals = [
        time
        for time in time_intervals
        if time >= lower_bound and pd.isna(time) == False
    ]

    return sorted(time_intervals)


def format_df_to_text(df):
    output = ""

    for row in df.iterrows():
        for column in df.columns:
            if column in ["subject_id", "hadm_id"]:
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
                            if str(v).lower() not in ["none", "nan", "nat"]:
                                output += f"   - {k}: {v}\n"
                else:
                    for v in value:
                        if str(v).lower() not in ["none", "nan", "nat"]:
                            output += f" - {v}\n"
            elif not isinstance(value, list) and str(value).lower() not in [
                "none",
                "nan",
                "nat",
            ]:
                output += f"{column}: {value}\n"

        output += "\n"

    return output
