from __future__ import annotations

import json
import logging
import os
import time
import h5py
import numpy as np
from MDAnalysis import Universe
import pathlib

import pendulum

from airflow import DAG
from airflow.models import Variable
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
CONCEPT = "hdf5_analysis"
UNIQUE_DAG_ID = f"{username}_{CONCEPT}"

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


def resolve_analysis_inputs(**context) -> None:
    dataset_dir = context["ti"].xcom_pull(task_ids="download_dataset", key="dataset_dir")
    if not dataset_dir:
        raise RuntimeError("dataset_dir not set")

    h5_path = ""
    ptraj_idx_path = ""
    for root, _dirs, files in os.walk(dataset_dir):
        for fn in files:
            p = os.path.join(root, fn)
            if fn == "transposed_trajectory.h5" and "hdf5" in os.path.relpath(root, dataset_dir).split(os.sep):
                h5_path = p
            elif fn == "ptraj_indices.in":
                ptraj_idx_path = p

    if not (h5_path and ptraj_idx_path):
        raise FileNotFoundError(f"Could not find transposed_trajectory.h5 and ptraj_indices.in in {dataset_dir}")

    out_dir = os.path.join(OUTPUT_ROOT, context["run_id"], "analysis")
    os.makedirs(out_dir, exist_ok=True)

    ti = context["ti"]
    ti.xcom_push(key="hdf5_path", value=h5_path)
    ti.xcom_push(key="ptraj_indices_path", value=ptraj_idx_path)
    ti.xcom_push(key="out_dir", value=out_dir)
    print(f"Resolved analysis inputs:\n  h5={h5_path}\n  ptraj_indices={ptraj_idx_path}\n  out_dir={out_dir}")


def compute_distances_from_indices(**context) -> None:
    ti = context["ti"]
    h5_path = ti.xcom_pull(task_ids="resolve_analysis_inputs", key="hdf5_path")
    ptraj_idx_path = ti.xcom_pull(task_ids="resolve_analysis_inputs", key="ptraj_indices_path")
    out_dir = ti.xcom_pull(task_ids="resolve_analysis_inputs", key="out_dir")

    # Parse pairs: lines "distance <label> <i> <j>"
    pairs = []
    with open(ptraj_idx_path, "r") as f:
        for line in f:
            t = line.strip().split()
            if len(t) == 4 and t[0].lower() == "distance":
                label, i, j = t[1], int(t[2]), int(t[3])
                pairs.append((label, i, j))

    needed = sorted({i for _, i, _ in pairs} | {j for _, _, j in pairs})

    with h5py.File(h5_path, "r") as hf:
        g = hf["run0"]
        pos = {idx: g[f"atom{idx}"][...] for idx in needed}

    labels = [label for (label, _, _) in pairs]
    n_frames = next(iter(pos.values())).shape[0]
    distances = {label: np.linalg.norm(pos[i] - pos[j], axis=1) for (label, i, j) in pairs}

    out_csv = os.path.join(out_dir, "distances.csv")
    with open(out_csv, "w") as f:
        f.write("frame," + ",".join(labels) + "\n")
        for fr in range(n_frames):
            f.write(str(fr) + "," + ",".join(f"{distances[label][fr]:.6f}" for label in labels) + "\n")

    ti.xcom_push(key="distances_csv", value=out_csv)
    print(f"Wrote {len(labels)} distances over {n_frames} frames -> {out_csv}")

def upload_results(**context) -> None:
    out_dir = context["ti"].xcom_pull(task_ids=context["params"].get("resolve_task_id", "resolve_analysis_inputs"), key="out_dir")
    if not out_dir:
        raise RuntimeError("out_dir not set")

    dataset_id = context['dag_run'].conf.get('dataset_id') or context['params']['dataset_id']
    if not dataset_id:
        raise ValueError("No dataset_id provided. Set Variable UNIFIED_DATASET_ID or pass it in conf/params.")

    refresh_token = Variable.get("lexis_refresh_token")
    session = LexisSessionOffline(refresh_token=refresh_token)
    irods = iRODS(session=session, suppress_print=False)

    # Create a new dataset for the results
    new_dataset_id = irods.create_empty_dataset(
        access="project",
        project=PROJECT_SHORTNAME,
        title=f"Results for {dataset_id}",
        description="Results uploaded by Airflow DAG",
    )
    print(f"Created new dataset {new_dataset_id} for results.")

    # Upload all files from out_dir to the new dataset
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            local_path = os.path.join(root, fn)
            relative_path = os.path.relpath(local_path, out_dir)
            irods.upload_file_to_dataset(
                access="project",
                project=PROJECT_SHORTNAME,
                dataset_id=new_dataset_id,
                local_filepath=local_path,
                dest_filepath=relative_path,
            )
            print(f"Uploaded {relative_path} to dataset {new_dataset_id}.")

    print(f"All results uploaded to dataset {new_dataset_id}.")


with DAG(
    dag_id=UNIQUE_DAG_ID,
    start_date=pendulum.datetime(2025, 9, 1, tz="UTC"),
    schedule=None,
    params={
        "dataset_id": "",  # pass the RESULTS dataset id from the first DAG
        "resolve_task_id": "resolve_analysis_inputs",
    },
    tags=["analysis", "workshop", CONCEPT, username],
) as analysis_dag:
    dl = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
    )
    resolve_analysis = PythonOperator(
        task_id="resolve_analysis_inputs",
        python_callable=resolve_analysis_inputs,
    )
    compute = PythonOperator(
        task_id="compute_distances",
        python_callable=compute_distances_from_indices,
    )
    upload_analysis = PythonOperator(
        task_id="upload_results",
        python_callable=upload_results,
    )

    dl >> resolve_analysis >> compute >> upload_analysis