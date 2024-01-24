import pickle

import pandas as pd
from transformers import DataCollatorForLanguageModeling


def reorder_columns(df, columns):
    return df[[column for column in columns if column in df.columns]]


def convert_icd_to_text(icd_list, icd_type, engine):
    # Prepare a CASE statement for ordering
    case_statement = "CASE "
    for index, item in icd_list.iterrows():
        code = item["icd_code"].strip()  # Remove leading/trailing spaces
        version = item["icd_version"]
        case_statement += (
            f"WHEN icd_code = '{code}' AND icd_version = {version} THEN {index} "
        )

    case_statement += "END"

    # Create a WHERE IN clause
    icd_conditions = ", ".join(
        [
            f"('{item[1]['icd_code'].strip()}', {item[1]['icd_version']})"
            for item in icd_list.iterrows()
        ]
    )
    sql_query = f"""
    SELECT long_title 
    FROM mimiciv.mimiciv_hosp.d_icd_{icd_type} 
    WHERE (icd_code, icd_version) IN ({icd_conditions})
    ORDER BY {case_statement};
    """

    # Execute the query
    return pd.read_sql(sql_query, engine)["long_title"].tolist()


def convert_lab_id_to_info(labs, engine):
    # Prepare a CASE statement for ordering
    case_statement = "CASE "
    for index, item in labs.iterrows():
        item_id = item["itemid"]
        case_statement += f"WHEN itemid = {item_id} THEN {index} "

    case_statement += "END"

    # Create a WHERE IN clause
    lab_conditions = ", ".join([str(item[1]["itemid"]) for item in labs.iterrows()])
    sql_query = f"""
    SELECT *
    FROM mimiciv.mimiciv_hosp.d_labitems
    WHERE itemid IN ({lab_conditions})
    ORDER BY {case_statement};
    """

    # Execute the query
    returned = pd.read_sql(sql_query, engine)

    return labs.merge(returned, on="itemid", how="outer").drop("itemid", axis=1)


def to_clean_records(dataframe):
    return dataframe.apply(lambda row: row.dropna().to_dict(), axis=1).tolist()


def pickle_to_tensor(x):
    deserialized = pickle.loads(bytes.fromhex(x.replace("\\x", "")))
    return deserialized


def transform_dataset_to_tensor(examples):
    examples["attention_mask"] = [
        pickle_to_tensor(x) for x in examples["attention_mask"]
    ]
    examples["input_ids"] = [pickle_to_tensor(x) for x in examples["input_ids"]]

    return examples


class BatchPaddedCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        batch = {"input_ids": [], "attention_mask": []}
        for example in examples:
            batch["input_ids"].append(example["input_ids"])
            batch["attention_mask"].append(example["attention_mask"])
        batch = self.tokenizer.pad(batch, return_tensors="pt", padding="longest")

        return batch
