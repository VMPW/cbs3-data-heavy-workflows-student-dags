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


PROJECT_SHORTNAME = "cbs3-ida"
# Standardized DAG ID: {username}_{concept}_{timestamp}
username = pathlib.Path(__file__).resolve().parent.name
CONCEPT = "ptraj"
UNIQUE_DAG_ID = f"3_{username}_{CONCEPT}"
DOWNLOAD_ROOT = f"/opt/airflow/data/downloads/{username}"
OUTPUT_ROOT = f"/opt/airflow/data/outputs/{username}"

def download_dataset(**context):
    dataset_id = context['dag_run'].conf.get('dataset_id') or context['params']['dataset_id']
    if not dataset_id:
        raise ValueError("No dataset_id provided. Set Variable UNIFIED_DATASET_ID or pass it in conf/params.")
    os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
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

def resolve_inputs(**context) -> None:
    dataset_dir = context["ti"].xcom_pull(task_ids="download_dataset", key="dataset_dir")
    if not dataset_dir:
        raise RuntimeError("dataset_dir not set")

    tops: list[str] = []
    trajs: list[str] = []
    ptraj_ins: list[str] = []

    def _pick_one(files: list[str]) -> str:
        files = sorted(files)
        if not files:
            return ""
        return files[0]

    for root, _dirs, files in os.walk(dataset_dir):
        for fn in files:
            p = os.path.join(root, fn)
            lower = fn.lower()
            if lower.endswith(('.top', '.prmtop')):
                tops.append(p)
            elif lower.endswith(('.traj', '.nc', '.mdcrd', '.crd')):
                trajs.append(p)
            elif lower == 'nmr_unoe_ptraj.in':
                ptraj_ins.append(p)

    top = _pick_one(tops)
    traj = _pick_one(trajs)
    ptraj_in = _pick_one(ptraj_ins)

    if not (top and traj and ptraj_in):
        raise FileNotFoundError(
            f"Missing inputs in {dataset_dir}. Found top={bool(top)}, traj={bool(traj)}, ptraj_in={bool(ptraj_in)}"
        )

    print(f"Resolved: top={top}\ntraj={traj}\ninput={ptraj_in}")
    out_dir = os.path.join(OUTPUT_ROOT, context["run_id"])  # keep outputs separate by run
    os.makedirs(out_dir, exist_ok=True)

    ti = context["ti"]
    ti.xcom_push(key="top", value=top)
    ti.xcom_push(key="traj", value=traj)
    ti.xcom_push(key="ptraj_in", value=ptraj_in)
    ti.xcom_push(key="out_dir", value=out_dir)


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
    resolve = PythonOperator(
        task_id="resolve_inputs",
        python_callable=resolve_inputs,
    )
    # Python
    run_cpptraj = BashOperator(
        task_id="run_cpptraj",
        bash_command="""
              set -euo pipefail

              # Move into the dataset directory resolved in the previous step
              cd "$DATASET_DIR"

              # Run cpptraj using the resolved topology and input file
              /tmp/cpptraj/bin/cpptraj \
                -p "$TOP" \
                -y "$TRAJ" \
                -i "$INFILE"
            """,
        env={
            # dataset_dir is produced by download_dataset
            "DATASET_DIR": "{{ ti.xcom_pull(task_ids='download_dataset', key='dataset_dir') }}",
            "TOP": "{{ ti.xcom_pull(task_ids='resolve_inputs', key='top') }}",
            "TRAJ": "{{ ti.xcom_pull(task_ids='resolve_inputs', key='traj') }}",
            "INFILE": "{{ ti.xcom_pull(task_ids='resolve_inputs', key='ptraj_in') }}",
        },
    )

    fetch_data >> resolve >> run_cpptraj
