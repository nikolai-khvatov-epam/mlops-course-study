#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd
import os
from prefect import flow, task, get_run_logger  # Added for orchestration

# Global model load (preserved, but wrapped in a task for flow)
def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

categorical = ['PULocationID', 'DOLocationID']  # Preserved

@task(log_prints=True)
def read_data_task(filename):
    """Preserved original read_data function as a task."""
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

@task(log_prints=True)
def predict_task(df, dv, model):
    """Logical block for prediction (preserved logic)."""
    dicts = df[categorical].to_dict(orient='records')
    numerical = ['trip_distance']  # Preserved
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['prediction'] = y_pred  # Preserved

    std_dev = df['prediction'].std()
    print(std_dev)

    mean_pred = df['prediction'].mean()
    print(f"Mean predicted duration: {mean_pred}")

    return df

@task(log_prints=True)
def save_task(df, year, month):
    """Logical block for saving (preserved logic)."""
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = df[['ride_id', 'prediction']]
    output_file = f'predictions_yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(size_mb)

@flow(name="Batch Inference Flow", log_prints=True)
def batch_inference_flow(year=2023, month=3):
    logger = get_run_logger()

    # Load model (preserved, now as part of flow)
    dv, model = load_model()

    # Input file (preserved logic, dynamic)
    input_file = f'./data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    # Run tasks (orchestrated)
    df = read_data_task(input_file)
    df = predict_task(df, dv, model)
    save_task(df, year, month)

if __name__ == '__main__':
    batch_inference_flow()