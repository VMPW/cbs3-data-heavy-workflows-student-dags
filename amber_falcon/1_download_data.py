from __future__ import annotations

import json
import os
import pathlib
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# py4lexis for dataset download
try:
    from py4lexis.core.session import LexisSessionOffline
    from py4lexis.core.lexis_irods import iRODS
except Exception:
    LexisSessionOffline = None  # type: ignore
    iRODS = None  # type: ignore

DOWNLOAD_ROOT = "/opt/airflow/data/downloads"
OUTPUT_ROOT = "/opt/airflow/data/outputs"
PROJECT_SHORTNAME = "cbs3-ida"
# Standardized DAG ID: {username}_{concept}_{timestamp}
username = pathlib.Path(__file__).resolve().parent.name
CONCEPT = "ptraj"
UNIQUE_DAG_ID = f"1_{username}_{CONCEPT}"

def download_dataset(**context):
    dataset_id = context['dag_run'].conf.get('dataset_id') or context['params']['dataset_id']
    if not dataset_id:
        raise ValueError("No dataset_id provided. Set Variable UNIFIED_DATASET_ID or pass it in conf/params.")

    # Create all necessary directories
    os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    out_dir = os.path.join(DOWNLOAD_ROOT, dataset_id)
    context["ti"].xcom_push(key="dataset_dir", value=out_dir)

    if os.path.exists(out_dir):
        print(f"Dataset {dataset_id} already present at {out_dir}; skipping download.")
        return

    refresh_token = Variable.get("lexis_refresh_token")
    session = LexisSessionOffline(refresh_token=refresh_token)
    irods = iRODS(session=session, suppress_print=False)

    print(f"Downloading dataset {dataset_id} to {DOWNLOAD_ROOT}...")
    irods.download_dataset_as_directory(access="project", project=PROJECT_SHORTNAME, dataset_id=dataset_id, local_directorypath=DOWNLOAD_ROOT)
    print("Download finished.")


with DAG(
    dag_id=UNIQUE_DAG_ID,
    start_date=pendulum.datetime(2025, 9, 1, tz="UTC"),
    schedule=None,
    params={
        "dataset_id": "",
    },
    tags=["simulation", "workshop", CONCEPT, username],
) as dag:
    fetch_data = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
    )

    fetch_data
