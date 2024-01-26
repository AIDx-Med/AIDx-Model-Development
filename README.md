<!---
Copyright 2024 The AIDx Team. All Rights reserved
--->

# AIDx Model Development

## Overview 📖
AIDx Model Development is an innovative repository focused on developing AI-driven models for healthcare applications. This project specializes in processing medical data, particularly from the MIMIC-IV database, and offers tools for model training and dataset management.

## Features 🚀
- **Data Processing**: Scripts for tokenizing and transforming healthcare data such as MIMIC-IV 🏥.
- **Model Training & Fine-Tuning**: Utilizes fine-tuning to create AI models on medical datasets 🧬.
- **Dataset Management**: Tools to create and handle datasets, including Parquet datasets, for healthcare AI 📊.

## Components 🛠
- `aidx.py`: The gateway to various functionalities like dataset creation and model tuning 🌐.
- **Docker Support**: Facilitates hosting the Jupyter Notebook environment 🐳.
- **Notebooks**: Step-by-step guides in Jupyter notebooks for data processing and analysis 📓.
- **Scripts**: Python scripts for comprehensive processes including database interactions and data workflows 📜.
- **Dependencies**: `requirements.txt` for easy installation of necessary packages 📌.

## Other Requirements
- Google Drive and `rclone` for dataset storage
    - Make sure the `rclone.conf` file is in the `scripts/utils` directory
- MIMIC-IV Database with PostgreSQL

## Getting Started 🚀
1. Clone the repository.
2. Set up the Docker environment, which hosts the Jupyter Notebook.
3. Install dependencies from `requirements.txt`.
4. Run `python aidx.py` to access the command menu and start exploring the functionalities.