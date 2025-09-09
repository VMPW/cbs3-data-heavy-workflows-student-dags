from __future__ import annotations

import json
import h5py
import logging
import os
import time
import numpy as np
import tqdm
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

PROJECT_SHORTNAME = "cbs3-ida"
# Standardized DAG ID: {username}_{concept}_{timestamp}
username = pathlib.Path(__file__).resolve().parent.name
CONCEPT = "hdf5_transposition"
UNIQUE_DAG_ID = f"5_{username}_{CONCEPT}"
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


def upload_results(**context) -> None:
    out_dir = context["ti"].xcom_pull(task_ids="resolve_inputs", key="out_dir")
    if not out_dir:
        raise RuntimeError("out_dir not set")

    dataset_id = context['dag_run'].conf.get('dataset_id') or context['params']['dataset_id']
    if not dataset_id:
        raise ValueError("No dataset_id provided. Set Variable UNIFIED_DATASET_ID or pass it in conf/params.")

    refresh_token = Variable.get("lexis_refresh_token")
    session = LexisSessionOffline(refresh_token=refresh_token)
    irods = iRODS(session=session, suppress_print=False)

    # Create a new dataset for the results
    response = irods.create_dataset(access="project", project=PROJECT_SHORTNAME, title=f"Results for {dataset_id}")
    new_dataset_id = response["dataset_id"]
    print(f"Created new dataset {new_dataset_id} for results.")

    # Upload all files from out_dir to the new dataset
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            local_path = os.path.join(root, fn)
            relative_path = os.path.relpath(local_path, out_dir)
            irods.put_data_object_to_dataset( dataset_id=new_dataset_id, local_filepath=local_path, dataset_filepath="./", access="project", project=PROJECT_SHORTNAME)
            print(f"Uploaded {relative_path} to dataset {new_dataset_id}.")

    print(f"All results uploaded to dataset {new_dataset_id}.")


def transpose_into_hdf5(**context) -> None:
    top = context["ti"].xcom_pull(task_ids="resolve_inputs", key="top")
    traj = context["ti"].xcom_pull(task_ids="resolve_inputs", key="traj")
    if not top or not traj:
        print("Inputs not resolved; skipping HDF5 transposition.")
        return
    out_root = context["ti"].xcom_pull(task_ids="resolve_inputs", key="out_dir")
    out_dir = os.path.join(out_root, "hdf5")
    os.makedirs(out_dir, exist_ok=True)

    u = Universe(top, traj, topology_format='TOP', format='TRJ')
    frame_limit = context['params'].get('frame_limit')
    atoms = u.select_atoms("all")
    n_atoms = len(atoms)
    n_frames = len(u.trajectory)
    if frame_limit and frame_limit < n_frames:
        n_frames = frame_limit
    out_path = os.path.join(out_dir, "transposed_trajectory.h5")
    t0 = time.perf_counter()

    logging.info(f"writing {n_frames} frames, {n_atoms} atoms -> {out_path}")
    with h5py.File(out_path, "w") as hf:
        run_group = hf.create_group("run0")
        dsets = [
            run_group.create_dataset(f"atom{aid}", (n_frames, 3), dtype="float32")
            for aid in range(n_atoms)
        ]
        # fill
        for i, ts in tqdm.tqdm(enumerate(u.trajectory)):
            if i >= n_frames:
                break
            pos = atoms.positions  # (n_atoms, 3), float64
            for aid in range(n_atoms):
                dsets[aid][i] = pos[aid]

    dt = time.perf_counter() - t0
    context["ti"].xcom_push(key="hdf5_path", value=out_path)
    context["ti"].xcom_push(key="prealloc_seconds", value=dt)
    print(f"[prealloc] done in {dt:.2f}s -> {out_path}")


def translate_ptraj_and_make_map(**context) -> None:
    """
    Make a simple JSON short-mask -> atom index map and a translated ptraj file with indices.
    - JSON: atom_index_map.json
    - Translated: ptraj_indices.in (lines: 'distance <label> <i> <j>')
    """
    ti = context["ti"]
    top = ti.xcom_pull(task_ids="resolve_inputs", key="top")
    ptraj_in = ti.xcom_pull(task_ids="resolve_inputs", key="ptraj_in")
    out_dir = ti.xcom_pull(task_ids="resolve_inputs", key="out_dir")

    u = Universe(top)
    atoms = u.atoms

    mask_to_idx = {}
    for i in range(len(atoms)):
        key = f":{int(atoms.resids[i])}@{atoms.names[i]}"
        if key not in mask_to_idx:
            mask_to_idx[key] = i

    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, "atom_index_map.json")
    with open(map_path, "w") as f:
        json.dump(mask_to_idx, f, indent=2)

    pairs = []
    with open(ptraj_in, "r") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 4 and t[0].lower() == "distance":
                label, m1, m2 = t[1], t[2], t[3]
                i = mask_to_idx[m1]
                j = mask_to_idx[m2]
                pairs.append((label, i, j))

    ptraj_idx_path = os.path.join(out_dir, "ptraj_indices.in")
    with open(ptraj_idx_path, "w") as f:
        for label, i, j in pairs:
            f.write(f"distance {label} {i} {j}\n")

    ti.xcom_push(key="atom_map_path", value=map_path)
    ti.xcom_push(key="ptraj_indices_path", value=ptraj_idx_path)
    print(f"[translate] wrote {map_path} and {ptraj_idx_path}")


with DAG(
    dag_id=UNIQUE_DAG_ID,
    start_date=pendulum.datetime(2025, 9, 1, tz="UTC"),
    schedule=None,
    params={
        "dataset_id": "",
        "frame_limit": None,
        "resolve_task_id": "resolve_inputs",  # used by upload_results to pick out_dir
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
    transpose = PythonOperator(
        task_id="transpose_into_hdf5",
        python_callable=transpose_into_hdf5,
    )
    translate = PythonOperator(
        task_id="translate_ptraj",
        python_callable=translate_ptraj_and_make_map,
    )
    upload = PythonOperator(
        task_id="upload_results",
        python_callable=upload_results,
    )

    # Parallel branches after resolve, then both join into upload
    fetch_data >> resolve
    resolve >> [transpose, translate] >> upload